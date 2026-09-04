-- ============================================================
-- AI Driver Safety & Intelligence Platform
-- Initial Database Schema Migration
-- Run this in the Supabase SQL Editor for your project.
-- ============================================================

-- ─── Extensions ──────────────────────────────────────────────
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================
-- TABLE: profiles
-- Mirrors auth.users; auto-created on signup via trigger.
-- ============================================================
CREATE TABLE IF NOT EXISTS public.profiles (
    id          UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email       TEXT NOT NULL,
    full_name   TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- ─── Trigger: auto-create profile on user signup ─────────────
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER SET search_path = public
AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name)
    VALUES (
        NEW.id,
        NEW.email,
        COALESCE(NEW.raw_user_meta_data->>'full_name', '')
    )
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ============================================================
-- TABLE: trips
-- One trip = one dashcam video upload + its analysis.
-- ============================================================
CREATE TABLE IF NOT EXISTS public.trips (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id      UUID NOT NULL REFERENCES public.profiles(id) ON DELETE CASCADE,
    title        TEXT NOT NULL,
    video_path   TEXT,                        -- Supabase Storage path
    duration     DOUBLE PRECISION,            -- seconds (set after processing)
    status       TEXT NOT NULL DEFAULT 'UPLOADED'
                     CHECK (status IN (
                         'UPLOADED',
                         'PROCESSING',
                         'ANALYZING',
                         'COMPLETED',
                         'FAILED'
                     )),
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_trips_user_id    ON public.trips(user_id);
CREATE INDEX IF NOT EXISTS idx_trips_status     ON public.trips(status);
CREATE INDEX IF NOT EXISTS idx_trips_created_at ON public.trips(created_at DESC);

-- ============================================================
-- TABLE: frames
-- Individual video frames extracted during processing.
-- ============================================================
CREATE TABLE IF NOT EXISTS public.frames (
    id           UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trip_id      UUID NOT NULL REFERENCES public.trips(id) ON DELETE CASCADE,
    frame_number INTEGER NOT NULL,
    timestamp    DOUBLE PRECISION NOT NULL,   -- seconds from video start
    image_path   TEXT                         -- Supabase Storage path
);

CREATE INDEX IF NOT EXISTS idx_frames_trip_id      ON public.frames(trip_id);
CREATE INDEX IF NOT EXISTS idx_frames_frame_number ON public.frames(trip_id, frame_number);

-- ============================================================
-- TABLE: detections
-- YOLO object detections per frame.
-- ============================================================
CREATE TABLE IF NOT EXISTS public.detections (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trip_id     UUID NOT NULL REFERENCES public.trips(id) ON DELETE CASCADE,
    frame_id    UUID REFERENCES public.frames(id) ON DELETE SET NULL,
    object_type TEXT NOT NULL,               -- car, pedestrian, bus, etc.
    confidence  DOUBLE PRECISION NOT NULL CHECK (confidence >= 0 AND confidence <= 1),
    x1          DOUBLE PRECISION NOT NULL,
    y1          DOUBLE PRECISION NOT NULL,
    x2          DOUBLE PRECISION NOT NULL,
    y2          DOUBLE PRECISION NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_detections_trip_id  ON public.detections(trip_id);
CREATE INDEX IF NOT EXISTS idx_detections_frame_id ON public.detections(frame_id);
CREATE INDEX IF NOT EXISTS idx_detections_type     ON public.detections(object_type);

-- ============================================================
-- TABLE: driving_metrics
-- Per-timestamp driving metrics (ESTIMATED / INFERRED values).
-- ============================================================
CREATE TABLE IF NOT EXISTS public.driving_metrics (
    id                   UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trip_id              UUID NOT NULL REFERENCES public.trips(id) ON DELETE CASCADE,
    timestamp            DOUBLE PRECISION NOT NULL,  -- seconds from video start
    speed                DOUBLE PRECISION,    -- km/h ESTIMATED (optical flow)
    acceleration         DOUBLE PRECISION,    -- m/s² ESTIMATED
    lane_deviation       DOUBLE PRECISION,    -- px ESTIMATED
    vehicle_density      INTEGER,             -- count INFERRED (YOLO)
    pedestrian_distance  DOUBLE PRECISION,    -- px ESTIMATED (YOLO bounding box)
    time_to_collision    DOUBLE PRECISION,    -- s ESTIMATED (heuristic)
    created_at           TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_metrics_trip_id   ON public.driving_metrics(trip_id);
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON public.driving_metrics(trip_id, timestamp);

-- ============================================================
-- TABLE: driving_events
-- Detected safety-relevant events.
-- ============================================================
CREATE TABLE IF NOT EXISTS public.driving_events (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trip_id     UUID NOT NULL REFERENCES public.trips(id) ON DELETE CASCADE,
    event_type  TEXT NOT NULL,
                -- e.g. PEDESTRIAN_PROXIMITY, LANE_DEPARTURE,
                --      HIGH_TRAFFIC_DENSITY, UNSAFE_FOLLOWING_DISTANCE
    severity    TEXT NOT NULL CHECK (severity IN ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL')),
    confidence  DOUBLE PRECISION CHECK (confidence >= 0 AND confidence <= 1),
    timestamp   DOUBLE PRECISION NOT NULL,   -- seconds from video start
    frame_id    UUID REFERENCES public.frames(id) ON DELETE SET NULL,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_events_trip_id   ON public.driving_events(trip_id);
CREATE INDEX IF NOT EXISTS idx_events_type      ON public.driving_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_severity  ON public.driving_events(severity);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON public.driving_events(trip_id, timestamp);

-- ============================================================
-- TABLE: risk_predictions
-- ANN / ML model risk outputs per timestamp.
-- ============================================================
CREATE TABLE IF NOT EXISTS public.risk_predictions (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trip_id       UUID NOT NULL REFERENCES public.trips(id) ON DELETE CASCADE,
    timestamp     DOUBLE PRECISION NOT NULL,
    risk_score    DOUBLE PRECISION NOT NULL CHECK (risk_score >= 0 AND risk_score <= 1),
    risk_class    TEXT NOT NULL CHECK (risk_class IN ('SAFE', 'MODERATE', 'HIGH')),
    model_name    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_risk_trip_id   ON public.risk_predictions(trip_id);
CREATE INDEX IF NOT EXISTS idx_risk_timestamp ON public.risk_predictions(trip_id, timestamp);

-- ============================================================
-- TABLE: model_versions
-- Registry of trained ML model versions & their metrics.
-- ============================================================
CREATE TABLE IF NOT EXISTS public.model_versions (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name  TEXT NOT NULL,
    model_type  TEXT NOT NULL,   -- CNN, ANN, XGBOOST, RANDOM_FOREST, etc.
    version     TEXT NOT NULL,
    metrics     JSONB,           -- {"accuracy": 0.92, "f1": 0.89, ...}
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (model_name, version)
);

CREATE INDEX IF NOT EXISTS idx_model_versions_name ON public.model_versions(model_name);

-- ============================================================
-- ROW LEVEL SECURITY
-- Every user can only access their own data.
-- ============================================================

-- profiles
ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own profile"
    ON public.profiles FOR SELECT
    USING (auth.uid() = id);

CREATE POLICY "Users can update own profile"
    ON public.profiles FOR UPDATE
    USING (auth.uid() = id);

-- trips
ALTER TABLE public.trips ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view own trips"
    ON public.trips FOR SELECT
    USING (auth.uid() = user_id);

CREATE POLICY "Users can insert own trips"
    ON public.trips FOR INSERT
    WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update own trips"
    ON public.trips FOR UPDATE
    USING (auth.uid() = user_id);

CREATE POLICY "Users can delete own trips"
    ON public.trips FOR DELETE
    USING (auth.uid() = user_id);

-- frames (access scoped via trips)
ALTER TABLE public.frames ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view frames of own trips"
    ON public.frames FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.trips
            WHERE trips.id = frames.trip_id
              AND trips.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert frames into own trips"
    ON public.frames FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.trips
            WHERE trips.id = frames.trip_id
              AND trips.user_id = auth.uid()
        )
    );

-- detections
ALTER TABLE public.detections ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view detections of own trips"
    ON public.detections FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.trips
            WHERE trips.id = detections.trip_id
              AND trips.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert detections into own trips"
    ON public.detections FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.trips
            WHERE trips.id = detections.trip_id
              AND trips.user_id = auth.uid()
        )
    );

-- driving_metrics
ALTER TABLE public.driving_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view metrics of own trips"
    ON public.driving_metrics FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.trips
            WHERE trips.id = driving_metrics.trip_id
              AND trips.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert metrics into own trips"
    ON public.driving_metrics FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.trips
            WHERE trips.id = driving_metrics.trip_id
              AND trips.user_id = auth.uid()
        )
    );

-- driving_events
ALTER TABLE public.driving_events ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view events of own trips"
    ON public.driving_events FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.trips
            WHERE trips.id = driving_events.trip_id
              AND trips.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert events into own trips"
    ON public.driving_events FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.trips
            WHERE trips.id = driving_events.trip_id
              AND trips.user_id = auth.uid()
        )
    );

-- risk_predictions
ALTER TABLE public.risk_predictions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view risk predictions of own trips"
    ON public.risk_predictions FOR SELECT
    USING (
        EXISTS (
            SELECT 1 FROM public.trips
            WHERE trips.id = risk_predictions.trip_id
              AND trips.user_id = auth.uid()
        )
    );

CREATE POLICY "Users can insert risk predictions into own trips"
    ON public.risk_predictions FOR INSERT
    WITH CHECK (
        EXISTS (
            SELECT 1 FROM public.trips
            WHERE trips.id = risk_predictions.trip_id
              AND trips.user_id = auth.uid()
        )
    );

-- model_versions (public read — no user scoping needed)
ALTER TABLE public.model_versions ENABLE ROW LEVEL SECURITY;

CREATE POLICY "model_versions readable by authenticated users"
    ON public.model_versions FOR SELECT
    TO authenticated
    USING (true);

-- ============================================================
-- STORAGE BUCKETS
-- Create via SQL (or manually in Supabase dashboard).
-- ============================================================

-- videos bucket
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'videos',
    'videos',
    false,
    524288000,   -- 500 MB per file
    ARRAY['video/mp4', 'video/quicktime', 'video/x-msvideo', 'video/avi', 'video/webm']
)
ON CONFLICT (id) DO NOTHING;

-- frames bucket
INSERT INTO storage.buckets (id, name, public, file_size_limit, allowed_mime_types)
VALUES (
    'frames',
    'frames',
    false,
    5242880,     -- 5 MB per frame image
    ARRAY['image/jpeg', 'image/png', 'image/webp']
)
ON CONFLICT (id) DO NOTHING;

-- model-artifacts bucket
INSERT INTO storage.buckets (id, name, public, file_size_limit)
VALUES (
    'model-artifacts',
    'model-artifacts',
    false,
    2147483648   -- 2 GB
)
ON CONFLICT (id) DO NOTHING;

-- ─── Storage RLS Policies ────────────────────────────────────

-- videos: users can read/write their own folder
CREATE POLICY "Users can upload own videos"
    ON storage.objects FOR INSERT TO authenticated
    WITH CHECK (
        bucket_id = 'videos'
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

CREATE POLICY "Users can view own videos"
    ON storage.objects FOR SELECT TO authenticated
    USING (
        bucket_id = 'videos'
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

CREATE POLICY "Users can delete own videos"
    ON storage.objects FOR DELETE TO authenticated
    USING (
        bucket_id = 'videos'
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

-- frames: users can read/write their own folder
CREATE POLICY "Users can upload own frames"
    ON storage.objects FOR INSERT TO authenticated
    WITH CHECK (
        bucket_id = 'frames'
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

CREATE POLICY "Users can view own frames"
    ON storage.objects FOR SELECT TO authenticated
    USING (
        bucket_id = 'frames'
        AND (storage.foldername(name))[1] = auth.uid()::text
    );

-- model-artifacts: authenticated read, service-role write only (via FastAPI backend)
CREATE POLICY "Authenticated users can read model artifacts"
    ON storage.objects FOR SELECT TO authenticated
    USING (bucket_id = 'model-artifacts');
