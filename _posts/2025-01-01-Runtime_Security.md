---
title: Runtime Security
date: 2025-01-01 21:15:00 +0800
categories: [Notes, OS]
tags: [security, os, notes]
---

### Linux x86 calling convention

int main() {
    int result = add(3, 5);
    return 0;
}

In the example above, main() is the caller, and add() is the callee.

- Callee-saved Registers
  - Registers whose values must be preserved by the called function (callee) across a function call. The callee must ensure that these registers have the same values when the function returns as they did when the function was entered.
  - If the callee uses these registers, it must save their values to the stack at the beginning of the function and restore them before returning.
  - These registers are typically used to preserve the caller's context, ensuring that the caller can continue using these registers after the function call.
  - In x86-64 architecture, rbx, rbp, and r12 to r15 are typical Callee-saved registers.
- Caller-saved
  - Registers whose values may be modified by the called function. The caller must assume that these registers' values will not be preserved across a function call.
  - If the caller wishes to retain the values of these registers after a function call, it must save them to the stack before calling the function and restore them afterward.
  - These registers are typically used for temporary storage, such as passing arguments between functions or storing intermediate results.

### Stack Frame

A stack frame is a block of memory allocated on the stack during a function call. It stores the function's local variables, parameters, return address, and other information related to the function call. Each function call creates a new stack frame. Therefore, a stack frame corresponds to an unfinished function.

### Safe and Unsafe Language

- Safe Language
  - Use reference rather than raw pointers.
  - Automatic tracking of data lifetimes.
- Unsafe Language
  - Allow raw pointers.
  - Manual tracking of data lifetimes.

### Buffer Overflow

  - Code Injection Attack: Overwrite the return address to point back to the buffer and executes the code they input into the buffer.
  - Code Reuse Attack: Overwrite the return address to point to existing code (e.g., a function in the program).

### Shadow Canary

- A canary value is placed between the data and the frame pointer on the stack.
- The value is randomly selected and checked before the function returns to ensure it hasn't been tampered with.
- If an attacker attempts to overwrite the return address, they must also overwrite the canary, which will be detected.
- Incremental Guessing can reduce the difficulty of making guesses.

### Data Execution Prevention (DEP)

- Enforces memory regions as either executable or writable, but not both (W^X).
  - Code segments: Read-only, executable.
  - Stack and heap: Writable, non-executable.
- Prevents code injection attacks.
- Just-In-Time (JIT) Compilers: These require memory to be both writable and executable, which can be a challenge for DEP.

### Shadow Stack

- Protects control flow integrity by maintaining a separate, secure stack for return addresses.
- The shadow stack mirrors the regular stack but is only used to store return addresses. Any attempt to modify the return address on the regular stack will be detected by comparing it with the shadow stack.

### Address Space Layout Randomization

- Randomizes the memory layout of a program's key areas (stack, heap, libraries, etc.) each time it is executed.
- Makes it more difficult for attackers to predict the location of specific functions or data.

### Control Flow Integrity (CFI)

CFI ensures that program execution follows the intended control flow, protecting against hijacking attacks.

- Forward Egde CFI
  - Protect control flow by verifying function calls.
- Backward Edge CFI
  - Protect control flow by verifying return addresses.

### Pointer Authentication Codes (PAC)

- Utilizes unused upper bits of pointers in 64-bit architectures to store an authentication code.
- Pointers are signed with a secret key, generating a PAC.
- The PAC is embedded in the pointer's upper bits.
- Before dereferencing, the pointer's PAC is verified to ensure integrity.

### Control Flow Guard (CFG)

- Prevent the misuse of indirect calls or jumps by ensuring they only transfer control to legitimate, pre-defined targets.
- Compile-Time Analysis:
  - The compiler identifies valid call and jump targets during the program's build process.
  - A control flow graph (CFG) is constructed, listing all permissible target addresses.
- Runtime Enforcement:
  - Before executing an indirect call or jump, the runtime verifies that the target address is part of the valid set.
  - Invalid targets cause the program to terminate or trigger an alert.