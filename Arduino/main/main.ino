#include <AccelStepper.h>

// Define pins
#define microPin1 2
#define microPin2 3
#define microPin3 4

#define stepPin_one 8
#define dirPin_one 9

#define stepPin_two 10
#define dirPin_two 11

long stepIncrement = 200; // Adjust this value as needed

// stepper accel objs
AccelStepper stepper_one(AccelStepper::DRIVER, stepPin_one, dirPin_one);
AccelStepper stepper_two(AccelStepper::DRIVER, stepPin_two, dirPin_two);

float speed_one = 0;
float speed_two = 0;
float baseSpeed = 5000; // steps per second

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
  
  // Setup stepper one
  stepper_one.enableOutputs();
  stepper_one.setMaxSpeed(10000);
  stepper_one.setAcceleration(5000);
  stepper_one.setCurrentPosition(0);

  
  Serial.println("Stepper one initialized");

  // Setup stepper two
  stepper_two.enableOutputs();
  stepper_two.setMaxSpeed(10000);
  stepper_two.setAcceleration(5000);
  stepper_two.setCurrentPosition(0);
 
  
  Serial.println("Stepper two initialized");  
    
}

void loop() {
  if (Serial.available() > 0) {
    char key = Serial.read();

    switch (key) {
      case 'w':
      case 'W':
        speed_one = baseSpeed;   // move forward
        Serial.println("Stepper one: UP");
        break;

      case 's':
      case 'S':
        speed_one = -baseSpeed;  // move backward
        Serial.println("Stepper one: DOWN");
        break;

      case 'a':
      case 'A':
        speed_two = -baseSpeed;  // move left
        Serial.println("Stepper two: LEFT");
        break;

      case 'd':
      case 'D':
        speed_two = baseSpeed;   // move right
        Serial.println("Stepper two: RIGHT");
        break;

      case 'x':
      case 'X':
        // stop both
        speed_one = 0;
        speed_two = 0;
        Serial.println("STOP");
        break;
    }
  }

  // Update speeds
  stepper_one.setSpeed(speed_one);
  stepper_two.setSpeed(speed_two);

  // Run both at constant speed
  stepper_one.runSpeed();
  stepper_two.runSpeed();
}