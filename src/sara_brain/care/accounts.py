"""Account management helpers for Sara Care setup."""

from __future__ import annotations

from ..core.brain import Brain
from ..models.neuron import NeuronType
from ..storage.account_repo import Account


def create_patient(brain: Brain, name: str, pin: str | None = None) -> Account:
    """Create the patient (reader) account and their concept neuron."""
    # Create the person as a concept neuron in Sara's knowledge
    neuron, _ = brain.neuron_repo.get_or_create(name.lower(), NeuronType.CONCEPT)

    # Teach Sara who this person is
    brain.teach(f"{name} is the patient", user_initiated=True)

    account = brain.account_repo.create(
        name=name, role="reader", pin=pin, neuron_id=neuron.id
    )
    brain.conn.commit()
    return account


def create_family(brain: Brain, name: str, relationship: str,
                  patient_name: str, pin: str | None = None) -> Account:
    """Create a family member (teacher) account."""
    neuron, _ = brain.neuron_repo.get_or_create(name.lower(), NeuronType.CONCEPT)

    # Teach Sara the relationship
    brain.teach(f"{name} is {patient_name}'s {relationship}", user_initiated=True)

    account = brain.account_repo.create(
        name=name, role="teacher", pin=pin, neuron_id=neuron.id
    )
    brain.conn.commit()
    return account


def create_doctor(brain: Brain, name: str, pin: str | None = None) -> Account:
    """Create a doctor/caregiver account."""
    neuron, _ = brain.neuron_repo.get_or_create(name.lower(), NeuronType.CONCEPT)

    brain.teach(f"{name} is a doctor", user_initiated=True)

    account = brain.account_repo.create(
        name=name, role="doctor", pin=pin, neuron_id=neuron.id
    )
    brain.conn.commit()
    return account


def setup_wizard(brain: Brain) -> list[Account]:
    """Interactive setup for creating accounts. Returns list of created accounts."""
    accounts: list[Account] = []

    print("\n  Sara Care — First Time Setup")
    print("  ============================\n")

    # Patient
    print("  Who is the person Sara will be helping?")
    patient_name = input("  Their name: ").strip()
    if not patient_name:
        print("  A name is required.")
        return []

    patient_pin = input("  Set a PIN for them (or press Enter for none): ").strip()
    patient = create_patient(brain, patient_name, patient_pin or None)
    accounts.append(patient)
    print(f"  Created account for {patient_name} (patient)\n")

    # Family members
    while True:
        print("  Add a family member? (or press Enter to skip)")
        family_name = input("  Their name: ").strip()
        if not family_name:
            break
        relationship = input(f"  {family_name}'s relationship to {patient_name} (e.g., son, daughter, spouse): ").strip()
        if not relationship:
            relationship = "family"
        family_pin = input("  Set a PIN (or press Enter for none): ").strip()
        family = create_family(brain, family_name, relationship, patient_name,
                               family_pin or None)
        accounts.append(family)
        print(f"  Created account for {family_name} ({relationship})\n")

    # Doctor
    print("  Add a doctor or caregiver? (or press Enter to skip)")
    doctor_name = input("  Their name: ").strip()
    if doctor_name:
        doctor_pin = input("  Set a PIN (or press Enter for none): ").strip()
        doctor = create_doctor(brain, doctor_name, doctor_pin or None)
        accounts.append(doctor)
        print(f"  Created account for {doctor_name} (doctor)\n")

    print(f"  Setup complete. {len(accounts)} accounts created.")
    return accounts


def select_account(brain: Brain) -> Account | None:
    """Interactive account selection at startup. Returns the selected account."""
    accounts = brain.account_repo.list_all()

    if not accounts:
        return None

    print("\n  Who am I speaking with?\n")
    for i, account in enumerate(accounts, 1):
        role_label = {
            "reader": "(that's me!)",
            "teacher": "(family)",
            "doctor": "(doctor)",
        }.get(account.role, "")
        print(f"  {i}. {account.name} {role_label}")
    print(f"  {len(accounts) + 1}. Someone new")

    while True:
        try:
            choice = input("\n  > ").strip()
            if not choice:
                continue
            idx = int(choice)
            if 1 <= idx <= len(accounts):
                account = accounts[idx - 1]
                # Verify PIN if set
                if account.pin_hash:
                    pin = input("  PIN: ").strip()
                    if not brain.account_repo.verify_pin(account.id, pin):
                        print("  Incorrect PIN.")
                        continue
                return account
            elif idx == len(accounts) + 1:
                return None  # Signal to create new account
            else:
                print("  Please choose a number from the list.")
        except ValueError:
            print("  Please enter a number.")
