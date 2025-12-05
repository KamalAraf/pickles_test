#include <Servo.h>

Servo mioServo;
int angolo = 90;           // posizione iniziale
const int step = 10;       // passo in gradi

void setup() {
  Serial.begin(9600);
  mioServo.attach(9);
  mioServo.write(angolo);
  Serial.println("Pronto: premi '1' per +10°, '2' per -10°");
  Serial.print("Angolo: ");
  Serial.println(angolo);
}

void loop() {
  while (Serial.available() > 0) {
    char c = Serial.read();

    if (c == '1') {
      angolo += step;
      if (angolo > 180) angolo = 180;
      mioServo.write(angolo);
      Serial.print("Aumentato -> ");
      Serial.println(angolo);
    }
    else if (c == '2') {
      angolo -= step;
      if (angolo < 0) angolo = 0;
      mioServo.write(angolo);
      Serial.print("Diminuìto -> ");
      Serial.println(angolo);
    }
  }
}
