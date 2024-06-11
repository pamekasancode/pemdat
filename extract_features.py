import cv2

def extract_features(image_path):
    # Baca gambar
    image = cv2.imread(image_path)
    
    # Ubah gambar ke skala abu-abu
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Thresholding untuk menghasilkan gambar biner
    _, binary_image = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    # Ekstraksi kontur
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Ambil kontur terbesar (asumsi raisin adalah objek terbesar)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Hitung momen geometri
    moments = cv2.moments(largest_contour)
    
    # Ekstraksi fitur-fitur
    area = moments['m00']
    major_axis_length = moments['mu20'] ** 0.5
    minor_axis_length = moments['mu02'] ** 0.5
    eccentricity = ((moments['mu20'] - moments['mu02']) ** 2 + 4 * moments['mu11'] ** 2) ** 0.5 / (moments['mu20'] + moments['mu02'])
    convex_area = cv2.contourArea(cv2.convexHull(largest_contour))
    extent = area / convex_area
    perimeter = cv2.arcLength(largest_contour, True)
    
    # Mengembalikan nilai fitur-fitur
    return {'area': area, 'major_axis_length': major_axis_length, 'minor_axis_length': minor_axis_length,
            'eccentricity': eccentricity, 'convex_area': convex_area, 'extent': extent, 'perimeter': perimeter}
