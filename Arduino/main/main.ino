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

# stepper accel objs
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
  stepper_one.setMaxSpeed(2000);
  stepper_one.setAcceleration(1280);
  stepper_one.setCurrentPosition(0);
  
  // Initial movement
  stepper_one.moveTo(5 * steps);
  
  Serial.println("Stepper one initialized");

  // Setup stepper two
  stepper_two.enableOutputs();
  stepper_two.setMaxSpeed(2000);
  stepper_two.setAcceleration(1280);
  stepper_two.setCurrentPosition(0);
  
  // Initial movement
  stepper_two.moveTo(5 * steps);
  
  Serial.println("Stepper two initialized");
    
}

void loop() {
  // Run the stepper (this must be called frequently)
  stepper_one.run();
  stepper_two.run();
  
  stepper_one.setSpeed(10000);
  stepper_two.setSpeed(10000);

  stepper_one.moveTo(5 * steps);
  stepper_two.moveTo(5 * steps);
  delay(500);

  }
}
