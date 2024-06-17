import sensor
import image
import lcd
import utime
import KPU as kpu
from Maix import GPIO
from board import board_info
from fpioa_manager import fm

# เริ่มต้นการใช้งาน LCD
lcd.init()
lcd.rotation(2)

# ตั้งค่า GPIO
fm.register(20, fm.fpioa.GPIOHS0, force=True)
outputvalueperson = GPIO(GPIO.GPIOHS0, GPIO.OUT)
outputvalueperson.value(0)

fm.register(21, fm.fpioa.GPIOHS1, force=True)
outputvaluelane = GPIO(GPIO.GPIOHS1, GPIO.OUT)
outputvaluelane.value(0)

fm.register(19, fm.fpioa.GPIOHS2, force=True)
outputvaluedoubleperson = GPIO(GPIO.GPIOHS2, GPIO.OUT)
outputvaluedoubleperson.value(0)

# ตั้งค่าเซ็นเซอร์
sensor.reset()
sensor.set_pixformat(sensor.RGB565)
sensor.set_framesize(sensor.QVGA)
sensor.set_vflip(0)
sensor.run(1)

# กำหนดเกณฑ์การตรวจจับสีเขียว
green_threshold = (43, 100, -88, 8, 15, 127)

# เริ่มต้นนาฬิกา
clock = utime.clock()

# กำหนดคลาสวัตถุที่ต้องการ
classes = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

# โหลดโมเดล YOLO2 และกำหนดค่า
task = kpu.load(0x500000)
anchor = (1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52)
kpu.init_yolo2(task, 0.5, 0.3, 5, anchor)

# ตัวแปรสำหรับบัฟเฟอร์การนับจำนวนคน
person_buffer = [0] * 10  # ความยาวของบัฟเฟอร์สำหรับค่าเฉลี่ย
double_person_buffer = [0] * 10  # บัฟเฟอร์สำหรับการตรวจจับคน 2 คน
buffer_length = 10
last_update_time = utime.time()  # เวลาที่ต้องหน่วง (เริ่มต้นที่ 0)
last_save_time = 0  # เวลาที่ล่าสุดที่บันทึกภาพ (เริ่มต้นที่ 0)
mun = 0  # เลขลำดับของภาพที่ถูกบันทึก
person_detected = False  # ตรวจจับเจอคนหรือไม่

# ฟังก์ชันสำหรับเพิ่มค่าในบัฟเฟอร์และคำนวณค่าเฉลี่ย
def update_buffer(buffer, value, max_length):
    if len(buffer) >= max_length:
        buffer.pop(0)
    buffer.append(value)
    return sum(buffer) / len(buffer)

# ฟังก์ชันสำหรับบันทึกภาพ
def save_image(img):
    global mun
    mun += 1
    img.save("/sd/" + str(mun) + ".jpg")
    print("Saved /sd/" + str(mun) + ".jpg")

# วนลูปเพื่อประมวลผลเฟรมภาพ
while True:
    # บันทึกเวลาของเฟรมภาพ
    clock.tick()

    # รับภาพใหม่จากเซ็นเซอร์
    img = sensor.snapshot()

    # ใช้โมเดล YOLO2 เพื่อตรวจจับวัตถุ
    objects = kpu.run_yolo2(task, img)

    # ตรวจจับสีเขียวในบริเวณกรอบที่กำหนด
    blobs = img.find_blobs([green_threshold], area_threshold=40, pixels_threshold=20, roi=(240,160,70,70))

    # หากพบสีเขียว
    if blobs:
        # กำหนดค่า GPIO สำหรับเลน
        outputvaluelane.value(1)
    else:
        # ปิด GPIO เลน
        outputvaluelane.value(0)

    person_count = 0  # รีเซ็ตตัวนับจำนวนคนในแต่ละเฟรม

    # ตรวจจับวัตถุจากผลการเรียก `kpu.run_yolo2()`
    if objects:
        for obj in objects:
            if classes[obj.classid()] == 'person':
                person_detected = True
                # วาดกรอบและข้อความเพื่อแสดงผลการตรวจจับคน
                img.draw_rectangle(obj.rect(), color=(0, 255, 0), thickness=3)
                img.draw_string(obj.x(), obj.y(), classes[obj.classid()], color=(255, 0, 0), scale=2)
                img.draw_string(obj.x(), obj.y() + 38, '%.3f' % obj.value(), color=(0, 0, 255), scale=2)
                break  # เมื่อเจอคนแล้วจะไม่ต้องตรวจจับวัตถุอื่น

    # เพิ่มค่าจำนวนคนในบัฟเฟอร์และคำนวณค่าเฉลี่ย
    avg_person_count = update_buffer(person_buffer, 1 if person_detected else 0, buffer_length)
    avg_double_person_count = update_buffer(double_person_buffer, 1 if avg_person_count >= 2 else 0, buffer_length)

    # ส่งค่า GPIO ทุกๆ 0.1 วินาที
    current_time = utime.time()
    if current_time - last_update_time >= 0.1:
        outputvalueperson.value(1 if avg_person_count > 0 else 0)
        outputvaluedoubleperson.value(1 if avg_double_person_count > 0 else 0)
        last_update_time = current_time

    # บันทึกภาพทุก 0.1 วินาที หากตรวจจับเจอคน
    if person_detected and current_time - last_save_time >= 0.1:
        save_image(img)
        last_save_time = current_time
        person_detected = False  # รีเซ็ตตัวแปรสำหรับการตรวจจับเจอคน

    # แสดงภาพบนหน้าจอ LCD
    img = img.resize(240, 240)
    lcd.display(img)

# เมื่อเสร็จสิ้นโปรแกรม ทำการ deinit โมเดล
kpu.deinit(task)
