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
    refutations INTEGER NOT NULL DEFAULT 0,
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

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_seg_source ON segments(source_id, strength DESC);
CREATE INDEX IF NOT EXISTS idx_seg_target ON segments(target_id);
CREATE INDEX IF NOT EXISTS idx_neuron_label ON neurons(label);
CREATE INDEX IF NOT EXISTS idx_neuron_type ON neurons(neuron_type);
CREATE INDEX IF NOT EXISTS idx_path_terminus ON paths(terminus_id);

-- Per-segment source provenance. A segment taught from two DIFFERENT
-- source_labels has two witnesses — the two-witness principle.
-- Re-teaching the same fact from the same source is a no-op (UNIQUE
-- constraint) and does NOT inflate the witness count.
CREATE TABLE IF NOT EXISTS segment_sources (
    id           INTEGER PRIMARY KEY,
    segment_id   INTEGER NOT NULL REFERENCES segments(id) ON DELETE CASCADE,
    source_label TEXT NOT NULL,
    created_at   REAL,
    UNIQUE(segment_id, source_label)
);
CREATE INDEX IF NOT EXISTS idx_seg_sources_segment ON segment_sources(segment_id);

-- Region registry — tracks which brain regions exist
CREATE TABLE IF NOT EXISTS regions (
    name        TEXT PRIMARY KEY,
    description TEXT,
    created_at  REAL
);

-- Cross-region bridges — explicit links between neurons in different regions
CREATE TABLE IF NOT EXISTS bridges (
    id              INTEGER PRIMARY KEY,
    source_region   TEXT NOT NULL,
    source_neuron_id INTEGER NOT NULL,
    target_region   TEXT NOT NULL,
    target_neuron_id INTEGER NOT NULL,
    relation        TEXT NOT NULL DEFAULT 'same_as',
    created_at      REAL,
    UNIQUE(source_region, source_neuron_id, target_region, target_neuron_id)
);
