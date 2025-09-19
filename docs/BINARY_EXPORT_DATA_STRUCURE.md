# Binary Data Structure Version 1

> **Note:**
>
> - All numbers are **little-endian**.
> - `n` indicates a previously read length.
> - `?` means variable size (compute from other definitions).

---

## **Export**

| Size | Type   | Description       |
| ---- | ------ | ----------------- |
| 9    | \_     | unknown/reserved  |
| 4    | `uint` | version           |
| ?    | Image  | embedded Image    |
| 8    | `uint` | number of patches |
| ?×n  | Patch  | n patches         |

---

## **Image**

| Size | Type   | Description   |
| ---- | ------ | ------------- |
| 2    | `uint` | width         |
| 2    | `uint` | height        |
| 1    | `bool` | raw           |
| 8    | `uint` | buffer length |
| n    | bytes  | buffer        |

---

## **4PTS (4 Points)**

| Size | Type        | Description          |
| ---- | ----------- | -------------------- |
| 64   | 4×[int,int] | 4 (x, y) coordinates |

---

## **TextBlock**

| Size | Type    | Description          |
| ---- | ------- | -------------------- |
| 8    | `uint`  | font size            |
| 8    | `float` | angle                |
| 8    | `float` | probability          |
| 1    | \_      | unknown/reserved     |
| 1    | `bool`  | fg_color available   |
| 0\|1 | `uint`  | fg_r (if available)  |
| 0\|1 | `uint`  | fg_g (if available)  |
| 0\|1 | `uint`  | fg_b (if available)  |
| 1    | `bool`  | bg_color available   |
| 0\|1 | `uint`  | bg_r (if available)  |
| 0\|1 | `uint`  | bg_g (if available)  |
| 0\|1 | `uint`  | bg_b (if available)  |
| 8    | `uint`  | original text length |
| n    | bytes   | original text        |
| 8    | `uint`  | 4PTS count           |
| n×64 | 4PTS    | 4PTS data            |

---

## **Patch**

| Size | Type      | Description        |
| ---- | --------- | ------------------ |
| 8    | `float`   | x                  |
| 8    | `float`   | y                  |
| ?    | Image     | embedded Image     |
| ?    | TextBlock | embedded TextBlock |

---
