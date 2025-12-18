let video;
let poseNet;
let pose;
let skeleton;

let brain;
let state = "waiting";
let targetLabel = 0;
let sampleCounter = 0;

function setup() {
  let canvas = createCanvas(640, 480);
  canvas.parent("cameraContainer");

  video = createCapture(VIDEO);
  video.size(640, 480);
  video.hide();

  poseNet = ml5.poseNet(video, () => {
    console.log("PoseNet Loaded");
  });

  poseNet.on("pose", function (results) {
    if (results.length > 0) {
      pose = results[0].pose;
      skeleton = results[0].skeleton;

      // 🔥 THIS LINE IS THE KEY
      if (state === "collecting") {
        recordKeypoints();
      }
    }
  });

  let options = {
    inputs: 34,
    outputs: 6,
    task: "classification",
    debug: true
  };

  brain = ml5.neuralNetwork(options);
}

function draw() {
  image(video, 0, 0);

  if (pose) {
    for (let kp of pose.keypoints) {
      fill(0, 255, 0);
      noStroke();
      circle(kp.position.x, kp.position.y, 8);
    }

    stroke(255);
    strokeWeight(2);
    for (let bone of skeleton) {
      line(
        bone[0].position.x,
        bone[0].position.y,
        bone[1].position.x,
        bone[1].position.y
      );
    }
  }
}

// ---------------- DATA COLLECTION ----------------

function startCollection(label) {
  targetLabel = label;
  state = "collecting";
  sampleCounter = 0;

  console.log("Started collecting Pose", label);
}

function stopCollection() {
  state = "waiting";
  console.log("Stopped. Samples collected:", sampleCounter);
}

function recordKeypoints() {
  if (!pose) return;

  let inputs = [];

  // ❌ NO FILTERS — RAW DATA
  for (let kp of pose.keypoints) {
    inputs.push(kp.position.x);
    inputs.push(kp.position.y);
  }

  let target = [targetLabel];

  brain.addData(inputs, target);
  sampleCounter++;

  // 🔥 VISUAL CONFIRMATION
  document.getElementById("statusText").innerText =
    `Collecting Pose ${targetLabel} | Samples: ${sampleCounter}`;
}

function saveData() {
  console.log("Saving dataset...");
  brain.saveData("basic");
}
