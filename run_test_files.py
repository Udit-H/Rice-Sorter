#!/usr/bin/env python3
"""Run validation checks on every file in Rice-Sorting/test."""

import os
import sys
import unittest

import cv2
import numpy as np

from rice_classifier_v4 import RiceClassifier, extract_features

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DIR = os.path.join(ROOT_DIR, "test")
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def get_test_files():
    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")
    for root, _, files in os.walk(TEST_DIR):
        for fname in sorted(files):
            yield os.path.join(root, fname)


def determine_feature_dim():
    sample_files = list(get_test_files())
    if not sample_files:
        raise ValueError(f"No test files found in {TEST_DIR}")
    sample_img = cv2.imread(sample_files[0])
    if sample_img is None:
        raise ValueError(f"Unable to load sample image: {sample_files[0]}")
    return extract_features(sample_img).shape[0]


def build_test_classifier():
    feature_dim = determine_feature_dim()
    clf = RiceClassifier(chalky_boost=1.4)
    rng = np.random.default_rng(42)
    X_train = rng.random((16, feature_dim), dtype=np.float32)
    y_train = np.array(["white", "chalky"] * 8)
    clf.fit(X_train, y_train)
    return clf, feature_dim


def print_classification_table(classifier, files):
    header = f"{'File':<44} {'Class':<10} {'Confidence':<12} {'Stage'}"
    sep = "=" * len(header)
    print(sep)
    print(header)
    print(sep)
    for file_path in files:
        if not os.path.isfile(file_path):
            continue
        result = classifier.predict_path(file_path)
        print(f"{os.path.basename(file_path):<44} {result['class']:<10} {result['confidence']:<12.4f} {result['stage']}")
    print(sep)
    print()


class TestRiceSortingFiles(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.classifier, cls.feature_dim = build_test_classifier()

    def test_all_files_exist_and_load(self):
        files = list(get_test_files())
        self.assertGreater(len(files), 0, f"No files found in {TEST_DIR}")

        for file_path in files:
            self.assertTrue(os.path.isfile(file_path), f"Missing file: {file_path}")
            ext = os.path.splitext(file_path)[1].lower()
            self.assertIn(ext, SUPPORTED_EXTENSIONS,
                          f"Unsupported file extension for test file: {file_path}")

            img = cv2.imread(file_path)
            self.assertIsNotNone(img, f"Unable to read image: {file_path}")
            self.assertGreater(img.size, 0, f"Image is empty: {file_path}")

    def test_extract_features_for_each_file(self):
        for file_path in get_test_files():
            img = cv2.imread(file_path)
            self.assertIsNotNone(img, f"Unable to read image: {file_path}")
            features = extract_features(img)
            self.assertIsInstance(features, np.ndarray,
                                  f"extract_features did not return ndarray for {file_path}")
            self.assertEqual(features.shape, (self.feature_dim,),
                             f"Unexpected feature vector shape for {file_path}: {features.shape}")

    def test_predict_path_for_each_file(self):
        for file_path in get_test_files():
            result = self.classifier.predict_path(file_path)
            self.assertIsInstance(result, dict,
                                  f"predict_path did not return dict for {file_path}")
            self.assertEqual(result.get("path"), file_path)
            self.assertIn(result.get("class"), {"white", "chalky", "yellow", "brown"})
            self.assertIsInstance(result.get("confidence"), float)
            self.assertGreaterEqual(result["confidence"], 0.0)
            self.assertLessEqual(result["confidence"], 1.0)
            self.assertIn(result.get("stage"), {"brown-rule", "yellow-rule", "white-chalky-ml"})


def main():
    files = list(get_test_files())
    classifier, _ = build_test_classifier()
    print("Classification results for test files:\n")
    print_classification_table(classifier, files)

    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestRiceSortingFiles)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
