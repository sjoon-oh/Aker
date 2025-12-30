-- Create default items table for vector rows

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE items (
    id SERIAL PRIMARY KEY,
    embedding vector(100) NOT NULL);

ALTER TABLE items ALTER COLUMN embedding SET STORAGE PLAIN;