#i#include <AccelStepper.h>

// Define pins
#define microPin1 2
#define microPin2 3
#define microPin4 4

#define dirPin_one 2
#define stepPin_one 3

#define dirPin_two 4
#define stepPin_two 5

unsigned long steps = 3000;

# Obj of stepper accel
AccelStepper stepper_one(AccelStepper::DRIVER, stepPin_one, dirPin_one);
AccelStepper stepper_two(AccelStepper::DRIVER, stepPin_two, dirPin_two);

void setup() {
  // Setup serial for debugging
  Serial.begin(9600);
  
  // Setup microstepping pins
  pinMode(microPin1, OUTPUT);
  pinMode(microPin2, OUTPUT);
  pinMode(microPin3, OUTPUT);
  
  // Set microstepping (HIGH,HIGH,HIGH = 1/8 step for most drivers)
  digitalWrite(microPin1, HIGH);
  digitalWrite(microPin2, HIGH);
  digitalWrite(microPin3, HIGH);
  
  // Setup stepper
  stepper.enableOutputs();
  stepper.setMaxSpeed(2000);
  stepper.setAcceleration(1280);
  stepper.setCurrentPosition(0);
  
  // Initial movement
  stepper.moveTo(5 * steps);
  
  Serial.println("Stepper initialized");
}

void loop() {
  // Run the stepper (this must be called frequently)
  stepper.run();
  
  // Check if we've reached the target position
  if (stepper.distanceToGo() == 0) {
    // Movement complete, reverse direction
    if (movingForward) {
      stepper.setSpeed(10000);

      stepper.moveTo(0);  // Move back to start
      movingForward = false;
      Serial.println("Moving backward");
    } else {
      stepper.setSpeed(10000);

      stepper.moveTo(5 * steps);  // Move forward
      movingForward = true;
      Serial.println("Moving forward");
    }
    
    // Small delay before next movement
    delay(500);
  }
}nclude <AccelStepper.h>

// Define pins
#define dirPin 2
#define stepPin 3
#define microPin3 5
#define microPin2 6
#define microPin1 7

unsigned long steps = 3000;
bool movingForward = true;

AccelStepper stepper(AccelStepper::DRIVER, stepPin, dirPin);

void setup() {
  // Setup serial for debugging
  Serial.begin(9600);
  
  // Setup microstepping pins
  pinMode(microPin1, OUTPUT);
  pinMode(microPin2, OUTPUT);
  pinMode(microPin3, OUTPUT);
  
  // Set microstepping (HIGH,HIGH,HIGH = 1/8 step for most drivers)
  digitalWrite(microPin1, HIGH);
  digitalWrite(microPin2, HIGH);
  digitalWrite(microPin3, HIGH);
  
  // Setup stepper
  stepper.enableOutputs();
  stepper.setMaxSpeed(2000);
  stepper.setAcceleration(1280);
  stepper.setCurrentPosition(0);
  
  // Initial movement
  stepper.moveTo(5 * steps);
  
  Serial.println("Stepper initialized");
}

void loop() {
  // Run the stepper (this must be called frequently)
  stepper.run();
  
  // Check if we've reached the target position
  if (stepper.distanceToGo() == 0) {
    // Movement complete, reverse direction
    if (movingForward) {
      stepper.setSpeed(10000);

      stepper.moveTo(0);  // Move back to start
      movingForward = false;
      Serial.println("Moving backward");
    } else {
      stepper.setSpeed(10000);

      stepper.moveTo(5 * steps);  // Move forward
      movingForward = true;
      Serial.println("Moving forward");
    }
    
    // Small delay before next movement
    delay(500);
  }
}
