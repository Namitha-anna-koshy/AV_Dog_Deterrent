import winsound
import time

def test_deterrent_frequencies():
    print("--- 🛡️ AV Dog Deterrent: Audio Hardware Test ---")
    
    # 1. Standard Test (1,000 Hz) - Everyone should hear this
    print("\n🔊 1. Playing Standard Beep (1,000 Hz)...")
    print("   (This sounds like a normal microwave beep)")
    winsound.Beep(1000, 1000) 
    time.sleep(1)

    # 2. High-Pitch Simulation (15,000 Hz) - Most young people can hear this
    print("\n🔊 2. Playing High-Pitch Simulation (15,000 Hz)...")
    print("   (This should sound like a very sharp, piercing whistle)")
    try:
        winsound.Beep(15000, 1000)
    except Exception as e:
        print(f"   ❌ Error: Your hardware or OS blocked 15kHz: {e}")
    time.sleep(1)

    # 3. Near-Ultrasonic (19,000 Hz) - Very difficult for humans to hear
    print("\n🔊 3. Playing Near-Ultrasonic (19,000 Hz)...")
    print("   (Most adults will hear SILENCE, but your speaker is working)")
    try:
        winsound.Beep(19000, 1000)
    except Exception as e:
        print(f"   ❌ Error: Your hardware cannot produce 19kHz: {e}")

    print("\n✅ Test Complete!")

if __name__ == "__main__":
    test_deterrent_frequencies()