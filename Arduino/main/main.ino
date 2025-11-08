#include <AccelStepper.h>

// Define pins
#define microPin1 2
#define microPin2 3
#define microPin3 4

#define stepPin_one 8
#define dirPin_one 9

#define stepPin_two 10
#define dirPin_two 11

long stepIncrement = 2000; // Adjust this value as needed

// stepper accel objs
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
    
    
    switch(key) {
      case 'w':
      case 'W':
        // Up arrow - move stepper one forward
        stepper_one.move(stepIncrement);
        Serial.println("Stepper one: UP");
        break;
        
      case 's':
      case 'S':
        // Down arrow - move stepper one backward
        stepper_one.move(-stepIncrement);
        Serial.println("Stepper one: DOWN");
        break;
        
      case 'a':
      case 'A':
        // Left arrow - move stepper two backward
        stepper_two.move(-stepIncrement);
        Serial.println("Stepper two: LEFT");
        break;
        
      case 'd':
      case 'D':
        // Right arrow - move stepper two forward
        stepper_two.move(stepIncrement);
        Serial.println("Stepper two: RIGHT");
        break;
    }
  }
  
  // Run both steppers
  stepper_one.run();
  stepper_two.run();



}
