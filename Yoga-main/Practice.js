let video;
let poseNet;
let pose;
let skeleton;
let brain;
let env;
let wave;

let posesArray = [
  "TADASANA",
  "ADHO MUKHO SAWASANA",
  "VIRABHADRASANA II",
  "VRIKSHASANA",
  "TRIKONASANA",
  "BHUJANGASANA"
];

let imgArray = [];
let poseCounter = 0;
let targetLabel = 1;

let iterationCounter = 0;
let missCounter = 0;
let timeLeft = 30;

// ---------------- SETUP ----------------

function setup() {
  let cam = createCanvas(320, 240);
  cam.parent("webcamContainer");

  env = loadSound("img/file.mp3");
  wave = loadSound("img/error.mp3");

  video = createCapture({
    video: {
      width: { ideal: 320 },
      height: { ideal: 240 },
      facingMode: "user"
    }
  });

  video.size(320, 240);
  video.hide();

  poseNet = ml5.poseNet(video, () => {
    console.log("PoseNet Ready");
  });
  poseNet.on("pose", gotPoses);

  // Pose images
  imgArray[0] = "img/Tadasana.jpeg";
  imgArray[1] = "img/Adho Mukho Sawasana.jpg";
  imgArray[2] = "img/Virabhadrasana II.jpg";
  imgArray[3] = "img/Vrikshasana.jpg";
  imgArray[4] = "img/Trikonasana.jpg";
  imgArray[5] = "img/Bhujangasana.jpg";

  document.getElementById("poseName").textContent = posesArray[0];
  document.getElementById("poseImg").src = imgArray[0];
  document.getElementById("time").textContent = "00:30";

  let options = {
    inputs: 34,
    outputs: 6,
    task: "classification",
    debug: true
  };

  brain = ml5.neuralNetwork(options);

  brain.load(
    {
      model: "model/model.json",
      metadata: "model/model_meta.json",
      weights: "model/model.weights.bin"
    },
    () => {
      console.log("Model Loaded");
      classifyPose();
    }
  );
}

// ---------------- POSE DETECTION ----------------

function gotPoses(poses) {
  if (poses.length > 0) {
    pose = poses[0].pose;
    skeleton = poses[0].skeleton;
  }
}

// ---------------- CLASSIFICATION ----------------

function classifyPose() {
  if (!pose) {
    setTimeout(classifyPose, 300);
    return;
  }

  let inputs = [];
  let baseX = pose.keypoints[0].position.x;
  let baseY = pose.keypoints[0].position.y;

  for (let kp of pose.keypoints) {
    if (kp.score > 0.5) {
      inputs.push(kp.position.x - baseX);
      inputs.push(kp.position.y - baseY);
    } else {
      inputs.push(0, 0);
    }
  }

  brain.classify(inputs, gotResult);
}

// ---------------- RESULT ----------------

function gotResult(error, results) {
  if (error || !results || results.length === 0) {
    setTimeout(classifyPose, 200);
    return;
  }

  let label = results[0].label;
  let confidence = results[0].confidence;

  console.log(
    "TARGET =", targetLabel,
    "| PRED =", label,
    "| CONF =", confidence.toFixed(2)
  );

  // DO NOTHING ELSE
  setTimeout(classifyPose, 500);
}


// ---------------- DRAW ----------------

function draw() {
  push();
  translate(width, 0);
  scale(-1, 1);
  image(video, 0, 0, width, height);

  if (pose) {
    for (let kp of pose.keypoints) {
      fill(0, 255, 0);
      noStroke();
      ellipse(kp.position.x, kp.position.y, 8);
    }

    for (let bone of skeleton) {
      stroke(255);
      strokeWeight(2);
      line(
        bone[0].position.x,
        bone[0].position.y,
        bone[1].position.x,
        bone[1].position.y
      );
    }
  }
  pop();
}

// ---------------- NEXT POSE ----------------

function nextPose() {
  iterationCounter = 0;
  missCounter = 0;
  timeLeft = 30;

  poseCounter++;

  if (poseCounter >= posesArray.length) {
    document.getElementById("welldone").textContent =
      "All poses completed!";
    document.getElementById("sparkles").style.display = "block";
    return;
  }

  targetLabel = poseCounter + 1;

  document.getElementById("poseName").textContent =
    posesArray[poseCounter];
  document.getElementById("poseImg").src =
    imgArray[poseCounter];
  document.getElementById("time").textContent = "00:30";

  setTimeout(classifyPose, 3000);
}
