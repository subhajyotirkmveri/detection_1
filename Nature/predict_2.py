from ultralytics import YOLO

model = YOLO('best.pt')

model.predict(source = 'test_12.jpeg', show = True, save = True, hide_conf = False, conf = 0.3, save_txt = False, save_crop = True, line_thickness = 1)
