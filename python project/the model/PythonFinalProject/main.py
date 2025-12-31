import os
import sys
import time


def main():
    while True:
        print("\n==========================================")
        print("   BLIND NAVIGATION ASSISTANCE SYSTEM     ")
        print("           Group 6 Launcher               ")
        print("==========================================")
        print("1. Launch Real-time Detection (Backend Core)")
        print("2. Launch UI Prototype (Frontend Design)")
        print("q. Quit")
        print("------------------------------------------")

        choice = input("Select an option (1/2/q): ")

        if choice == '1':
            print("\nStarting Real-time Detection System...")
            print("Loading Model... Please wait.")
            os.system('python run_final.py')

        elif choice == '2':
            print("\nStarting User Interface Prototype...")
            os.system('python ui_design.py')
            print("Error: Please update main.py with the teammate's filename.")

        elif choice.lower() == 'q':
            print("Exiting program.")
            sys.exit()
        else:
            print("\nInvalid selection. Please try again.")

if __name__ == "__main__":
    main()