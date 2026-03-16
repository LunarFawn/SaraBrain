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
    id          INTEGER PRIMARY KEY,
    origin_id   INTEGER NOT NULL REFERENCES neurons(id),
    terminus_id INTEGER NOT NULL REFERENCES neurons(id),
    source_text TEXT,
    created_at  REAL
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

CREATE INDEX IF NOT EXISTS idx_seg_source ON segments(source_id, strength DESC);
CREATE INDEX IF NOT EXISTS idx_seg_target ON segments(target_id);
CREATE INDEX IF NOT EXISTS idx_neuron_label ON neurons(label);
CREATE INDEX IF NOT EXISTS idx_neuron_type ON neurons(neuron_type);
CREATE INDEX IF NOT EXISTS idx_path_terminus ON paths(terminus_id);
