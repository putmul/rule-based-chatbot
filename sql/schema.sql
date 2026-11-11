CREATE DATABASE IF NOT EXISTS siagabot
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE siagabot;

CREATE TABLE IF NOT EXISTS chat_history (
    id BIGINT UNSIGNED AUTO_INCREMENT PRIMARY KEY,
    session_id VARCHAR(100) NOT NULL,
    user_message TEXT NOT NULL,
    bot_response TEXT NOT NULL,
    matched_intent VARCHAR(100) NULL,
    similarity_score DECIMAL(6,5) NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    INDEX idx_session_id (session_id),
    INDEX idx_created_at (created_at),
    INDEX idx_matched_intent (matched_intent)
);
