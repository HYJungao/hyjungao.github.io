---
title: Container
date: 2025-01-01 21:15:00 +0800
categories: [Notes, OS]
tags: [security, os, notes]
---

- Containers share the host's kernel and cannot have a different operating system or kernel than the host.

### Namespaces

Provide isolation by limiting what processes can see and interact with in their system view.

- pid
  - Processes within a PID namespace only see processes in the same namespace.
  - Each namespace has its own numbering scheme for process IDs.
- net
  - Processes within a given network namespace get their own private network stack, including: network interfaces, routing tables, iptables rules, sockets
- mnt
  - Processes can have their own root filesystem and "private" mounts.
  - Mounts can be private or shared.
- uts
  - Isolates hostname and NIS domain name, enabling containers to have their own hostname.
- ipc
  - Allows a group of processes to have their own: ipc semaphores, message queues, shared memory
- user
  - Allow to map UID/GID
    - UID 0 -> 1999 in container is mapped to UID 10000 -> 11999 on host.

### Capability
  - Allow fine-grained control over root privileges by dividing the root user's powers into distinct units instead of running a process as root.
  - Can drop unnecessary capabilities to minimize attack surfaces, reducing the risk of privilege escalation.

### Cgroups (Control Groups)
  - Organize and manage processes hierarchically.
  - Control and monitor resource usage (CPU, memory, disk I/O, etc.).
  - Resource Limiting: Restrict the amount of resources a process can use.
  - Resource Prioritization: Prioritize resource usage for critical processes.
  - Accounting: Collect usage data for monitoring.

### SELinux
  - A mechanism for supporting access control security policies.
  - Adds a mandatory access control (MAC) layer.
  - Can use SELinux policies to restrict access to resources and processes.

### Seccomp (Secure Computing Mode)

A kernel feature to restrict the system calls a process can make.

- Strict Mode: Allows only basic system calls (e.g., read, write, exit).
- Filter Mode: Uses Berkeley Packet Filter (BPF) to create fine-grained filters for system calls.
- Disabled Mode: No restrictions (default).

### Union File System (Overlay)

Achieve Copy-on-Write (CoW).

- Base (read-only): Unmodified files remain here.
- Upper (read-write): Modified or new files are stored here.
- New files are created only in the upper layer.
- Modified files are copied from lower to upper layers before changes are applied.
- Deleted files are "hidden" in upper layers but still exist in lower layers.

### Create a new root

- Create a mountpoint.
- Create a new mount namespace.
- Mount the container's filesystem.
- Chroot or pivot_root
