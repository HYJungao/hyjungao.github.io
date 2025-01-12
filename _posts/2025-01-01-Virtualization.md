---
title: Virtualization
date: 2025-01-01 21:25:00 +0800
categories: [Notes, OS]
tags: [security, os, notes]
---

### Types of Virtualization
- Type 1 (Baremetal)
  - The Virtual Machine Monitor (VMM) manages all hardware resources and supports the execution of Virtual Machines (VMs).
  - The hypervisor incorporates various drivers for device management. To streamline this, a privileged Service VM handles device management, reusing existing drivers, while the hypervisor manages CPU and memory resources.
  - When a guest OS issues an I/O request, the VMM forwards it to the Service VM, incurring context-switching overhead.
- Type 2 (Hosted)
  - The Host OS owns and manages all hardware resources.
  - The VMM operates as a module within the Host OS.

### Virtualization Techniques

- Full virtualization
  - The guest OS runs without modification.
  - The hypervisor emulates hardware behavior for the guest OS.
- Paravirtualization
  - The guest OS is modified to recognize it is running in a virtualized environment.
  - Explicit calls to the hypervisor (e.g., hypercalls, similar to syscalls) are used for privileged operations.

### x86 Architecture and Virtualization

- x86 has four protection levels (rings), with Ring 0 being the highest privilege level.
  - Root-Ring 0: Hypervisor operates here.
  - Non-Root-Ring 0: Guest OS operates here.
- When the guest OS attempts to perform privileged operations, a VMexit trap is triggered, switching control to the hypervisor in root mode.
- After completing the operation, the hypervisor performs a VMentry to return control to the VM.

### Instruction Handling

- Non-privileged instructions: Executed directly by the hardware.
- Privileged operations (trap and emulate):
  - Trap to the hypervisor.
  - If the operation is illegal, terminate the VM.
  - If legal, emulate the expected hardware behavior.

### Challenges in Early x86 Virtualization

- In early x86 architectures, certain privileged instructions did not trigger traps to the hypervisor when executed outside Ring 0 (the highest privilege level). Instead, these instructions would fail silently without notifying the hypervisor.
- This behavior created a significant issue: the hypervisor remained unaware that such instructions were executed, and thus took no action to modify the hardware state.
- Meanwhile, the guest OS assumed the privileged operation had completed successfully, leading to inconsistencies between the expected and actual hardware state.

### Dynamic Binary Translation

- Instruction sequences that are about the be executed are dynamically captured from the VM library.
- Checks for problematic instructions:
  - If none are found, allows native execution.
  - If problematic instructions are present, translates them into safe sequences or emulates the desired behavior. In some cases, the system may also emulate the desired behavior of the original instructions, potentially bypassing the need for a trap to the hypervisor.
- Optimizations of the translation overhead:
  - Cached translated blocks
  - Only analyze kernel code.

### Memory Virtualization

- Full Virtualization
  - Guest OS expects contiguous physical memory starting from 0x0.
  - Still able to use MMU and TLB
  - Address types:
    - Virtual Address (VA): Application-level.
    - Physical Address (PA): OS-level.
    - Machine Address (MA): Hypervisor-level.
  - Inefficiency: Guest page tables map VA → PA, while hypervisor maps PA → MA. Only the hypervisor can use the TLB for actual address translation.
  - Optimization: Use shadow page tables where the guest page tables map VA → PA and the hypervisor directly maps VA → MA.
- Paravirtualization
  - No requirement for contiguous physical memory.
  - Guest OS explicitly registers page tables with the hypervisor.
  - Batch page table updates to reduce traps to the hypervisor.

### Device Virtualization

- Passthrough:
  - VMM level dirver configures device access permissions.
  - VM has exclusive access to the device (no sharing).
  - VM directly accesses the device, bypassing the VMM.
  - Drawback: Device binding to the VM complicates migration.
- Hypervisor Direct:
  - VMM intercepts all device access requests.
  - Emulates device operations by translating them into generic I/O operations and invoking VMM-resident drivers.
  - Drawback: Introduces latency and increases hypervisor complexity due to diverse drivers.
- Split Device Driver:
  - Front-end driver: Runs in the guest VM.
  - Back-end driver: Runs in the Service VM or Host OS.
  - Guest drivers are modified to send I/O requests as messages to the Service VM.
  - Limitation: Restricted to paravirtualized guest OS.

### Hardware Virtualization Support

- VM Control Structure:
  - Per-VCPU configuration to specify whether system calls should trap.
- Extended Page Tables and Tagged TLB:
  - Tagged TLBs with VM IDs to avoid flushes during context switches.
- Multiqueue Devices:
  - Devices with multiple logical interfaces, each assignable to a separate VM.

### Security Issues in Virtualization

- DMA Attacks:
  - One VM can use DMA to overwrite another VM's memory.
  - Solution: Implement an I/O MMU to enforce access control for DMA operations.
- VMM Access to VM Memory
  - The VMM can read a VM's memory, posing a security risk.
  - Solution: Use transparent memory encryption/authentication for VM memory.