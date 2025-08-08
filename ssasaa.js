<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Pose Overlay with p5.js</title>
  <script src="https://cdnjs.cloudflare.com/ajax/libs/p5.js/1.6.0/p5.min.js"></script>
</head>
<body>
<script>
let video;
let poseData;
let overlayX = 0, overlayY = 0;
let dragging = false;
let dragOffsetX = 0, dragOffsetY = 0;

function preload() {
  // Load video and JSON data
  video = createVideo('hookayuda.mp4');
  video.hide();
  poseData = loadJSON('pose_data.json');
}

function setup() {
  createCanvas(800, 600);
  video.loop();
}

function draw() {
  background(0);
  image(video, 0, 0, width, height);

  if (poseData && poseData.frames && poseData.frames.length > 0) {
    let frameIndex = int((video.time() / video.duration()) * poseData.frames.length);
    frameIndex = constrain(frameIndex, 0, poseData.frames.length - 1);
    drawPose(poseData.frames[frameIndex]);
  }
}

function drawPose(frame) {
  push();
  translate(overlayX, overlayY);
  noFill();
  stroke(0, 255, 0);
  strokeWeight(3);

  // Draw points
  for (let p of frame.keypoints) {
    ellipse(p.x, p.y, 10, 10);
  }

  // Draw edges if available
  if (frame.edges) {
    for (let e of frame.edges) {
      let a = frame.keypoints[e[0]];
      let b = frame.keypoints[e[1]];
      line(a.x, a.y, b.x, b.y);
    }
  }
  pop();
}

function mousePressed() {
  dragging = true;
  dragOffsetX = mouseX - overlayX;
  dragOffsetY = mouseY - overlayY;
}

function mouseReleased() {
  dragging = false;
}

function mouseDragged() {
  if (dragging) {
    overlayX = mouseX - dragOffsetX;
    overlayY = mouseY - dragOffsetY;
  }
}
</script>
</body>
</html>
