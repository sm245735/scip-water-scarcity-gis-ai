CREATE EXTENSION IF NOT EXISTS postgis;
CREATE SCHEMA IF NOT EXISTS recovery;
CREATE TABLE IF NOT EXISTS recovery.sources (
    sha256 text PRIMARY KEY, -- identity hash: relative path + content SHA256
    content_sha256 text NOT NULL,
    path text NOT NULL,
    kind text NOT NULL,
    bytes bigint NOT NULL,
    rows integer NOT NULL,
    imported_at timestamptz NOT NULL DEFAULT now()
);
-- Immutable source versions. Row numbers are data row ordinals, excluding headers.
CREATE TABLE IF NOT EXISTS recovery.raw_records (
    source_sha256 text REFERENCES recovery.sources,
    row_number integer NOT NULL,
    payload jsonb NOT NULL,
    PRIMARY KEY (source_sha256, row_number)
);
CREATE TABLE IF NOT EXISTS recovery.weather (
    source_sha256 text REFERENCES recovery.sources,
    row_number integer NOT NULL,
    stno text NOT NULL,
    obs_date date NOT NULL,
    source_priority integer NOT NULL,
    pp01 double precision, tx01 double precision, tx02 double precision,
    rh01 double precision, wd01 double precision, ps01 double precision,
    quality jsonb NOT NULL,
    PRIMARY KEY (source_sha256, row_number)
);
CREATE INDEX IF NOT EXISTS weather_station_date ON recovery.weather(stno, obs_date);
CREATE TABLE IF NOT EXISTS recovery.parse_rejections (
    source_sha256 text REFERENCES recovery.sources,
    row_number integer NOT NULL,
    reason text NOT NULL,
    PRIMARY KEY(source_sha256,row_number)
);
CREATE TABLE IF NOT EXISTS recovery.stations (
    stno text PRIMARY KEY, stname text NOT NULL,
    opened date, closed date,
    geom geometry(Point,4326), raw jsonb NOT NULL
);
-- Source selection is whole-row, not silent per-field mixing. Monthly CSV wins
-- over the old aggregate. Hash/row ordering breaks ties deterministically;
-- overlapping disagreements are exported for review.
CREATE OR REPLACE VIEW recovery.weather_daily AS
SELECT DISTINCT ON (stno,obs_date) * FROM recovery.weather
ORDER BY stno,obs_date,source_priority DESC,source_sha256,row_number;
COMMENT ON VIEW recovery.weather_daily IS
'Reconstructed conservative data, not a restoration of the original June v2 table.';
