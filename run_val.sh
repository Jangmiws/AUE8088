# python val.py --weights runs/train/yolov5n-rgbt/weights/best.pt --data data/kaist-rgbt.yaml --name yolov5n-rgbt-results --imgsz 640 --task val --save-json --rgbt
# python val.py --weights runs/train/yolov5n-rgbt-aug/weights/best.pt --data data/kaist-rgbt.yaml --name yolov5n-rgbt-aug-results --imgsz 640 --task val --save-json --rgbt
# python val.py --weights runs/train/yolov5n-rgbt-aug-anchor/weights/best.pt --data data/kaist-rgbt.yaml --name yolov5n-rgbt-aug-anchor-results --imgsz 640 --task val --save-json --rgbt
# python val.py --weights runs/train/yolov5n-rgbt-aug2-anchor/weights/best.pt --data data/kaist-rgbt.yaml --name yolov5n-rgbt-aug2-anchor-loss-results --imgsz 640 --task val --save-json --rgbt
# python val.py --weights runs/train/yolov5n-rgbt-aug-anchor-100/weights/best.pt --data data/kaist-rgbt.yaml --name yolov5n-rgbt-aug-anchor-100-results --imgsz 640 --task val --save-json --rgbt
# python val.py --weights runs/train/yolov5n-rgbt-aug2-anchor-100/weights/best.pt --data data/kaist-rgbt.yaml --name yolov5n-rgbt-aug2-anchor100-results --imgsz 640 --task val --save-json --rgbt
# python val.py --weights runs/train/yolov5n-rgbt-aug3-anchor-100/weights/best.pt --data data/kaist-rgbt.yaml --name yolov5n-rgbt-aug3-anchor100-results --imgsz 640 --task val --save-json --rgbt
# python val.py --weights runs/train/yolov5s-rgbt-aug-anchor/weight/best.pt --data data/kaist-rgbt.yaml --name yolov5s-rgbt-aug-anchor-results --imgsz 640 --task val --save-json --rgbt
python val.py --weights runs/train/yolov5n-rgbt-aug2-anchor-val/weights/best.pt --data data/kaist-rgbt.yaml --name yolov5n-rgbt-aug2-anchor-val-results --imgsz 640 --task val --save-json --rgbt
