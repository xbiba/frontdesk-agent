-- Local development only. Production roles and secrets come from infrastructure code (M5).
-- The application connects as frontdesk_app, which is subject to row-level security.
CREATE ROLE frontdesk_app LOGIN PASSWORD 'frontdesk_app' NOSUPERUSER NOCREATEDB NOCREATEROLE NOBYPASSRLS;
CREATE DATABASE frontdesk_clinic OWNER frontdesk;
CREATE DATABASE frontdesk_shop OWNER frontdesk;
