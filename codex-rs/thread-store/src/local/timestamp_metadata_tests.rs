//! Verifies narrow SQLite timestamp updates preserve metadata and repair fallbacks.

use super::test_support::test_config;
use super::test_support::write_session_file_with_history_mode;
use super::*;
use chrono::Utc;
use codex_protocol::protocol::ThreadMemoryMode;
use pretty_assertions::assert_eq;
use std::sync::Arc;

#[tokio::test]
async fn timestamp_updates_repair_missing_rows_then_touch_only_timestamp_columns() {
    for history_mode in [ThreadHistoryMode::Legacy, ThreadHistoryMode::Paginated] {
        let home = tempfile::TempDir::new().expect("temp dir");
        let config = test_config(home.path());
        let runtime = codex_state::StateRuntime::init(
            config.sqlite.clone(),
            config.default_model_provider_id.clone(),
        )
        .await
        .expect("initialize state db");
        let pool = config
            .sqlite
            .open_read_write_pool(&config.sqlite.state_db_path())
            .await
            .expect("open SQL observer");
        let store = Arc::new(LocalThreadStore::new(config, Some(runtime.clone())));
        let id = uuid::Uuid::new_v4();
        let thread_id = ThreadId::from_string(&id.to_string()).expect("thread id");
        let path = write_session_file_with_history_mode(
            home.path(),
            "2025-01-03T12-00-00",
            id,
            history_mode,
        )
        .expect("write canonical history");
        let params = UpdateThreadMetadataParams {
            thread_id,
            patch: ThreadMetadataPatch {
                updated_at: Some(Utc::now()),
                ..Default::default()
            },
            include_archived: false,
        };
        // Missing rows must still be reconstructed from their canonical history.
        store
            .update_thread_metadata(params.clone())
            .await
            .expect("reconstruct missing metadata");
        let mut expected = runtime
            .get_thread(thread_id)
            .await
            .expect("read repaired metadata")
            .expect("repaired row");
        assert_eq!(
            (&expected.rollout_path, expected.history_mode),
            (&path, history_mode)
        );
        sqlx::raw_sql(
            r#"
            CREATE TABLE timestamp_writes(kind TEXT);
            CREATE TRIGGER count_timestamp AFTER UPDATE OF updated_at_ms ON threads
            BEGIN INSERT INTO timestamp_writes VALUES ('timestamp'); END;
            CREATE TRIGGER count_full_row AFTER UPDATE OF title ON threads
            BEGIN INSERT INTO timestamp_writes VALUES ('full'); END;
            "#,
        )
        .execute(&pool)
        .await
        .expect("install write counters");

        store
            .update_thread_metadata(params.clone())
            .await
            .expect("touch existing metadata");
        let actual = runtime
            .get_thread(thread_id)
            .await
            .expect("read touched metadata")
            .expect("touched row");
        assert!(actual.updated_at > expected.updated_at);
        expected.updated_at = actual.updated_at;
        assert_eq!(actual, expected);
        let writes: Vec<String> =
            sqlx::query_scalar("SELECT kind FROM timestamp_writes ORDER BY rowid")
                .fetch_all(&pool)
                .await
                .expect("read write counters");
        assert_eq!(writes, vec!["timestamp"]);

        store
            .update_thread_metadata(UpdateThreadMetadataParams {
                patch: ThreadMetadataPatch {
                    updated_at: Some(Utc::now()),
                    model_provider: Some(expected.model_provider.clone()),
                    title: Some(expected.title.clone()),
                    token_usage: Some(codex_protocol::protocol::TokenUsage {
                        total_tokens: expected.tokens_used,
                        ..Default::default()
                    }),
                    ..Default::default()
                },
                ..params.clone()
            })
            .await
            .expect("observe unchanged metadata facts");
        let writes: Vec<String> =
            sqlx::query_scalar("SELECT kind FROM timestamp_writes ORDER BY rowid")
                .fetch_all(&pool)
                .await
                .expect("read write counters");
        assert_eq!(writes, vec!["timestamp", "timestamp"]);

        store
            .stage_pending_thread_metadata(
                thread_id,
                ThreadMetadataPatch {
                    model: Some("staged-model".to_string()),
                    ..Default::default()
                },
            )
            .await
            .expect("stage metadata");
        store
            .update_thread_metadata(params)
            .await
            .expect("apply staged metadata");
        let actual = runtime
            .get_thread(thread_id)
            .await
            .expect("read staged metadata")
            .expect("updated row");
        expected.model = Some("staged-model".to_string());
        expected.updated_at = actual.updated_at;
        assert_eq!(actual, expected);
        assert!(
            store
                .pending_thread_metadata
                .lock(thread_id)
                .await
                .is_none()
        );
        let mut writes: Vec<String> = sqlx::query_scalar("SELECT kind FROM timestamp_writes")
            .fetch_all(&pool)
            .await
            .expect("read write counters");
        writes.sort();
        assert_eq!(writes, vec!["full", "timestamp", "timestamp", "timestamp"]);

        let resume_params = ResumeThreadParams {
            thread_id,
            rollout_path: Some(path.clone()),
            history: None,
            history_revision: None,
            include_archived: false,
            metadata: crate::ThreadPersistenceMetadata {
                cwd: Some(home.path().to_path_buf()),
                model_provider: expected.model_provider.clone(),
                memory_mode: ThreadMemoryMode::Enabled,
            },
        };
        let mut sync = crate::thread_metadata_sync::ThreadMetadataSync::for_resume(
            &resume_params,
            Some(&actual),
        );
        let token_count = |total_tokens| {
            RolloutItem::EventMsg(codex_protocol::protocol::EventMsg::TokenCount(
                codex_protocol::protocol::TokenCountEvent {
                    info: Some(codex_protocol::protocol::TokenUsageInfo {
                        total_token_usage: codex_protocol::protocol::TokenUsage {
                            total_tokens,
                            ..Default::default()
                        },
                        last_token_usage: Default::default(),
                        model_context_window: None,
                    }),
                    rate_limits: None,
                },
            ))
        };
        let delegated_output = |text| {
            RolloutItem::EventMsg(codex_protocol::protocol::EventMsg::ItemCompleted(
                codex_protocol::protocol::ItemCompletedEvent {
                    thread_id,
                    turn_id: "turn".to_string(),
                    item: codex_protocol::items::TurnItem::FunctionCallOutput(
                        codex_protocol::items::FunctionCallOutputItem {
                            id: "output".to_string(),
                            name: "create_thread".to_string(),
                            namespace: Some("codex_tui".to_string()),
                            output: codex_protocol::models::FunctionCallOutputBody::Text(format!(
                                "<codex_delegation>\n<source_thread_id>source</source_thread_id>\n<input>{text}</input>\n</codex_delegation>"
                            )),
                        },
                    ),
                    started_at_ms: Some(0),
                    completed_at_ms: 0,
                },
            ))
        };
        let mut previous_writes: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM timestamp_writes")
            .fetch_one(&pool)
            .await
            .expect("count prior writes");
        let creates_preview = actual.preview.as_deref().is_none_or(str::is_empty);
        let mut observations = vec![
            (token_count(42), true, 42),
            (token_count(42), false, 42),
            (token_count(43), true, 43),
        ];
        if creates_preview {
            observations.push((delegated_output("first task"), true, 43));
        }
        observations.push((delegated_output("later task"), false, 43));
        for (item, changes_metadata, total_tokens) in observations {
            let update = sync.observe_appended_items(&[item]);
            if let Some(update) = update.as_ref() {
                store
                    .update_thread_metadata(UpdateThreadMetadataParams {
                        thread_id,
                        patch: update.patch.clone(),
                        include_archived: false,
                    })
                    .await
                    .expect("persist observed token total");
            }
            let current_writes: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM timestamp_writes")
                .fetch_one(&pool)
                .await
                .expect("count observed writes");
            assert_eq!(current_writes > previous_writes, changes_metadata);
            assert_eq!(
                runtime
                    .get_thread(thread_id)
                    .await
                    .expect("read observed tokens")
                    .expect("thread metadata")
                    .tokens_used,
                total_tokens
            );
            previous_writes = current_writes;
            if let Some(update) = update.as_ref() {
                sync.mark_pending_update_applied(update);
            }
        }

        let older = sync
            .observe_appended_items(&[token_count(44)])
            .expect("older changed token total");
        let newer = sync
            .observe_appended_items(&[token_count(45)])
            .expect("newer changed token total");
        for update in [&older, &newer] {
            store
                .update_thread_metadata(UpdateThreadMetadataParams {
                    thread_id,
                    patch: update.patch.clone(),
                    include_archived: false,
                })
                .await
                .expect("persist outstanding token total");
        }
        sync.mark_pending_update_applied(&newer);
        sync.mark_pending_update_applied(&older);
        let update = sync
            .observe_appended_items(&[token_count(44)])
            .expect("a late completion must not hide a changed stored total");
        store
            .update_thread_metadata(UpdateThreadMetadataParams {
                thread_id,
                patch: update.patch,
                include_archived: false,
            })
            .await
            .expect("persist total after late completion");
        assert_eq!(
            runtime
                .get_thread(thread_id)
                .await
                .expect("read total after late completion")
                .expect("thread metadata")
                .tokens_used,
            44
        );

        let (live, _) = crate::LiveThread::resume(store, history_mode, resume_params)
            .await
            .expect("resume live thread for physical path repair");
        live.append_items(&[token_count(42)])
            .await
            .expect("record live token usage");
        live.flush()
            .await
            .expect("flush live usage before path repair");
        let parent = path.parent().expect("rollout parent");
        let alternate_path = parent
            .join("..")
            .join(parent.file_name().expect("rollout directory name"))
            .join(path.file_name().expect("rollout file name"));
        live.update_metadata(
            ThreadMetadataPatch {
                rollout_path: Some(alternate_path),
                ..Default::default()
            },
            false,
        )
        .await
        .expect("set an equivalent physical path spelling");
        let mut expected_repair = runtime
            .get_thread(thread_id)
            .await
            .expect("read metadata before physical repair")
            .expect("existing thread metadata");
        assert_ne!(expected_repair.rollout_path, path);
        expected_repair.rollout_path = path;
        sqlx::raw_sql(
            r#"
            CREATE TABLE path_repair_token_writes(kind TEXT);
            CREATE TRIGGER count_path_repair_tokens AFTER UPDATE OF tokens_used ON threads
            BEGIN INSERT INTO path_repair_token_writes VALUES ('tokens'); END;
            "#,
        )
        .execute(&pool)
        .await
        .expect("observe token writes during physical path repair");
        live.flush().await.expect("repair live rollout path");
        let actual_repair = runtime
            .get_thread(thread_id)
            .await
            .expect("read repaired physical path")
            .expect("existing repaired metadata");
        let token_writes: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM path_repair_token_writes")
            .fetch_one(&pool)
            .await
            .expect("count token writes during physical path repair");
        assert_eq!((actual_repair, token_writes), (expected_repair, 0));
        live.shutdown().await.expect("shutdown path repair fixture");
        pool.close().await;
    }
}
