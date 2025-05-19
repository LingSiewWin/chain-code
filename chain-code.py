import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_shape(shape='rectangle', size=200):
    img = np.ones((size, size), dtype=np.uint8) * 255
    if shape == 'rectangle':
        cv2.rectangle(img, (50, 50), (150, 150), 0, -1)
    elif shape == 'triangle':
        pts = np.array([[100, 30], [30, 170], [170, 170]], np.int32)
        cv2.drawContours(img, [pts], 0, 0, -1)
    elif shape == 'circle':
        cv2.circle(img, (100, 100), 60, 0, -1)
    return img

def get_chain_code(contour):
    direction_map = {
        (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
        (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7
    }
    code = []
    for i in range(1, len(contour)):
        dx = contour[i][0][0] - contour[i - 1][0][0]
        dy = contour[i][0][1] - contour[i - 1][0][1]
        direction = direction_map.get((dx, dy))
        if direction is not None:
            code.append(direction)
    return code

def visualize_chain_code(binary_img, contour, code):
    vis = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)
    reverse_map = {
        0: (1, 0), 1: (1, -1), 2: (0, -1), 3: (-1, -1),
        4: (-1, 0), 5: (-1, 1), 6: (0, 1), 7: (1, 1)
    }
    for i in range(1, len(contour)):
        x, y = contour[i - 1][0]
        direction = code[i - 1]
        dx, dy = reverse_map[direction]
        vis = cv2.arrowedLine(vis, (x, y), (x + dx*4, y + dy*4), (0, 0, 255), 1, tipLength=0.3)
    return vis

def main():
    # === User input ===
    shape = input("Choose shape (rectangle / triangle / circle): ").strip().lower()
    if shape not in ['rectangle', 'triangle', 'circle']:
        print("Invalid shape! Defaulting to rectangle.")
        shape = 'rectangle'

    # === Generate and process image ===
    img = generate_shape(shape)
    cv2.imwrite("image.png", img)

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        print("No contours found!")
        return

    contour = contours[0]
    chain_code = get_chain_code(contour)

    # === Visualization ===
    result_img = visualize_chain_code(binary, contour, chain_code)
    cv2.imwrite("chain_code_result.png", result_img)

    print(f"\n📌 Chain code length: {len(chain_code)}")
    print(f"📜 Chain code: {chain_code[:50]}{'...' if len(chain_code) > 50 else ''}")
    print("✅ Saved visualization as 'chain_code_result.png'")

    # === Show with matplotlib ===
    plt.figure(figsize=(6, 6))
    plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
    plt.title("Chain Code Visualization")
    plt.axis("off")
    plt.show()

if __name__ == "__main__":
    main()
