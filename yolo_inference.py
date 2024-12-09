from ultralytics import YOLO

############## Predicting  ##############

# model = YOLO("models/best.pt")


# result = model.predict(
#     "input_videos/input_video.mp4", conf=0.2, save=True, project="runs", name="predict"
# )

# print(result)

# print("boxes:")
# for box in result[0].boxes:
#     print(box)


############## Tracking  ##############


model = YOLO("yolov8x")


result = model.track(
    "input_videos/input_video.mp4", conf=0.2, save=True, project="runs", name="predict"
)
