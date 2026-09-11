export interface Detection {
    detection_id: string;
    class_id: number;
    class_name: string;
    ai_confidence: number;
    bbox_xyxy_pixel: number[];
}

export interface ModelData {
    frame_id: string;
    model_version: string;
    inference_latency_ms: number;
    detections: Detection[];
}

export interface FrameResult {
    frame_id: string;
    image_width: number;
    image_height: number;
    models: {
        crab_pot: ModelData;
        cylinder: ModelData;
    };
}

export interface AIDataResponse {
    pipeline_version: string;
    frames: FrameResult[];
}