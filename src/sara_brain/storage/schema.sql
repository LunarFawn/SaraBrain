CREATE TABLE IF NOT EXISTS neurons (
    id          INTEGER PRIMARY KEY,
    label       TEXT NOT NULL UNIQUE,
    neuron_type TEXT NOT NULL,
    created_at  REAL,
    metadata    TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id          INTEGER PRIMARY KEY,
    source_id   INTEGER NOT NULL REFERENCES neurons(id),
    target_id   INTEGER NOT NULL REFERENCES neurons(id),
    relation    TEXT NOT NULL,
    strength    REAL NOT NULL DEFAULT 1.0,
    traversals  INTEGER NOT NULL DEFAULT 0,
    created_at  REAL,
    last_used   REAL,
    UNIQUE(source_id, target_id, relation)
);

CREATE TABLE IF NOT EXISTS paths (
    id               INTEGER PRIMARY KEY,
    origin_id        INTEGER NOT NULL REFERENCES neurons(id),
    terminus_id      INTEGER NOT NULL REFERENCES neurons(id),
    source_text      TEXT,
    created_at       REAL,
    account_id       INTEGER REFERENCES accounts(id),
    trust_status     TEXT,
    repetition_count INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS path_steps (
    id          INTEGER PRIMARY KEY,
    path_id     INTEGER NOT NULL REFERENCES paths(id),
    step_order  INTEGER NOT NULL,
    segment_id  INTEGER NOT NULL REFERENCES segments(id),
    UNIQUE(path_id, step_order)
);

CREATE TABLE IF NOT EXISTS similarities (
    neuron_a_id INTEGER NOT NULL REFERENCES neurons(id),
    neuron_b_id INTEGER NOT NULL REFERENCES neurons(id),
    shared_paths INTEGER NOT NULL,
    overlap_ratio REAL NOT NULL,
    created_at  REAL,
    PRIMARY KEY (neuron_a_id, neuron_b_id)
);

CREATE TABLE IF NOT EXISTS associations (
    id             INTEGER PRIMARY KEY,
    association    TEXT NOT NULL,
    property_label TEXT NOT NULL,
    neuron_id      INTEGER REFERENCES neurons(id),
    created_at     REAL,
    UNIQUE(association, property_label)
);

CREATE TABLE IF NOT EXISTS question_words (
    association    TEXT PRIMARY KEY,
    question_word  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS categories (
    label    TEXT PRIMARY KEY,
    category TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS accounts (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    role        TEXT NOT NULL,
    pin_hash    TEXT,
    neuron_id   INTEGER REFERENCES neurons(id),
    created_at  REAL,
    is_active   INTEGER NOT NULL DEFAULT 1
);

CREATE TABLE IF NOT EXISTS interactions (
    id               INTEGER PRIMARY KEY,
    account_id       INTEGER NOT NULL REFERENCES accounts(id),
    interaction_type TEXT NOT NULL,
    content          TEXT NOT NULL,
    response         TEXT,
    path_ids         TEXT,
    created_at       REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_interaction_account ON interactions(account_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_interaction_type ON interactions(interaction_type, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_interaction_time ON interactions(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_path_trust ON paths(trust_status);
CREATE INDEX IF NOT EXISTS idx_path_account ON paths(account_id);
CREATE INDEX IF NOT EXISTS idx_seg_source ON segments(source_id, strength DESC);
CREATE INDEX IF NOT EXISTS idx_seg_target ON segments(target_id);
CREATE INDEX IF NOT EXISTS idx_neuron_label ON neurons(label);
CREATE INDEX IF NOT EXISTS idx_neuron_type ON neurons(neuron_type);
CREATE INDEX IF NOT EXISTS idx_path_terminus ON paths(terminus_id);
