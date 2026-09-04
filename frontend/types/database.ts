/**
 * Auto-generated Supabase database type definitions.
 * These match the schema in supabase/migrations/001_initial_schema.sql.
 *
 * In a real project, generate this with:
 *   npx supabase gen types typescript --project-id <project-id> > types/database.ts
 */

export type Json =
  | string
  | number
  | boolean
  | null
  | { [key: string]: Json | undefined }
  | Json[];

export type TripStatus = 'UPLOADED' | 'PROCESSING' | 'ANALYZING' | 'COMPLETED' | 'FAILED';
export type RiskClass  = 'SAFE' | 'MODERATE' | 'HIGH';
export type Severity   = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL';
export type ModelType  = 'CNN' | 'ANN' | 'XGBOOST' | 'RANDOM_FOREST' | 'TRANSFER_LEARNING' | 'YOLO';

export interface Database {
  public: {
    Tables: {
      profiles: {
        Row: {
          id:         string;
          email:      string;
          full_name:  string | null;
          created_at: string;
        };
        Insert: {
          id:         string;
          email:      string;
          full_name?: string | null;
          created_at?: string;
        };
        Update: {
          email?:     string;
          full_name?: string | null;
        };
      };

      trips: {
        Row: {
          id:           string;
          user_id:      string;
          title:        string;
          video_path:   string | null;
          duration:     number | null;
          status:       TripStatus;
          created_at:   string;
          completed_at: string | null;
        };
        Insert: {
          id?:          string;
          user_id:      string;
          title:        string;
          video_path?:  string | null;
          duration?:    number | null;
          status?:      TripStatus;
          created_at?:  string;
          completed_at?: string | null;
        };
        Update: {
          title?:        string;
          video_path?:   string | null;
          duration?:     number | null;
          status?:       TripStatus;
          completed_at?: string | null;
        };
      };

      frames: {
        Row: {
          id:           string;
          trip_id:      string;
          frame_number: number;
          timestamp:    number;
          image_path:   string | null;
        };
        Insert: {
          id?:          string;
          trip_id:      string;
          frame_number: number;
          timestamp:    number;
          image_path?:  string | null;
        };
        Update: {
          image_path?: string | null;
        };
      };

      detections: {
        Row: {
          id:          string;
          trip_id:     string;
          frame_id:    string | null;
          object_type: string;
          confidence:  number;
          x1:          number;
          y1:          number;
          x2:          number;
          y2:          number;
          created_at:  string;
        };
        Insert: {
          id?:         string;
          trip_id:     string;
          frame_id?:   string | null;
          object_type: string;
          confidence:  number;
          x1:          number;
          y1:          number;
          x2:          number;
          y2:          number;
          created_at?: string;
        };
        Update: Record<string, never>;
      };

      driving_metrics: {
        Row: {
          id:                  string;
          trip_id:             string;
          timestamp:           number;
          speed:               number | null;
          acceleration:        number | null;
          lane_deviation:      number | null;
          vehicle_density:     number | null;
          pedestrian_distance: number | null;
          time_to_collision:   number | null;
          created_at:          string;
        };
        Insert: {
          id?:                  string;
          trip_id:              string;
          timestamp:            number;
          speed?:               number | null;
          acceleration?:        number | null;
          lane_deviation?:      number | null;
          vehicle_density?:     number | null;
          pedestrian_distance?: number | null;
          time_to_collision?:   number | null;
          created_at?:          string;
        };
        Update: Record<string, never>;
      };

      driving_events: {
        Row: {
          id:          string;
          trip_id:     string;
          event_type:  string;
          severity:    Severity;
          confidence:  number | null;
          timestamp:   number;
          frame_id:    string | null;
          description: string | null;
          created_at:  string;
        };
        Insert: {
          id?:          string;
          trip_id:      string;
          event_type:   string;
          severity:     Severity;
          confidence?:  number | null;
          timestamp:    number;
          frame_id?:    string | null;
          description?: string | null;
          created_at?:  string;
        };
        Update: Record<string, never>;
      };

      risk_predictions: {
        Row: {
          id:            string;
          trip_id:       string;
          timestamp:     number;
          risk_score:    number;
          risk_class:    RiskClass;
          model_name:    string;
          model_version: string;
          created_at:    string;
        };
        Insert: {
          id?:           string;
          trip_id:       string;
          timestamp:     number;
          risk_score:    number;
          risk_class:    RiskClass;
          model_name:    string;
          model_version: string;
          created_at?:   string;
        };
        Update: Record<string, never>;
      };

      model_versions: {
        Row: {
          id:         string;
          model_name: string;
          model_type: string;
          version:    string;
          metrics:    Json | null;
          created_at: string;
        };
        Insert: {
          id?:         string;
          model_name:  string;
          model_type:  string;
          version:     string;
          metrics?:    Json | null;
          created_at?: string;
        };
        Update: {
          metrics?: Json | null;
        };
      };
    };

    Views:   Record<string, never>;
    Functions: Record<string, never>;
    Enums:   Record<string, never>;
  };
}
