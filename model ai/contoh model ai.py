"""
SISTEM DETEKSI TUBERCULOSIS DARI X-RAY
Program utama untuk mendeteksi TB dari gambar X-ray dada
"""

import os
import sys

# Ensure UTF-8 output encoding on Windows consoles
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
from datetime import datetime
import argparse
from PIL import Image
import matplotlib.pyplot as plt


def find_default_model():
    """Find available trained model file or folder"""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    workspace_dir = os.path.dirname(base_dir)
    candidates = [
        os.path.join(workspace_dir, "best_tbnet_model.keras"),
        os.path.join(workspace_dir, "best_tbnet_model.h5"),
        os.path.join(workspace_dir, "tbnet_output", "tbnet_tuberculosis_model.keras"),
        os.path.join(workspace_dir, "tbnet_output", "tbnet_tuberculosis_model.h5"),
        os.path.join(workspace_dir, "TuberculosisNet", "tbnet_model_ready.keras"),
        os.path.join(workspace_dir, "TuberculosisNet", "tbnet_model_ready.h5"),
        os.path.join(base_dir, "models", "TB-Net"),
        os.path.join(workspace_dir, "models", "TB-Net"),
        "best_tbnet_model.keras",
        "best_tbnet_model.h5",
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    return None


class TBDetectionSystem:
    """Sistem Deteksi Tuberculosis"""
    
    def __init__(self, model_path=None):
        """
        Inisialisasi sistem deteksi TB
        
        Args:
            model_path: Path ke file model Keras (.keras/.h5) atau folder checkpoint TB-Net
        """
        print("🔬 Memulai Sistem Deteksi Tuberculosis...")
        
        if model_path is None or not os.path.exists(model_path):
            found_path = find_default_model()
            if found_path is not None:
                model_path = found_path
            elif model_path is not None and not os.path.exists(model_path):
                print(f"❌ Model tidak ditemukan di: {model_path}")
                print("📥 Silakan letakkan model TB-Net (best_tbnet_model.keras / .h5) terlebih dahulu")
                sys.exit(1)
            else:
                print("❌ Model TB-Net tidak ditemukan.")
                print("📥 Pastikan file best_tbnet_model.keras atau best_tbnet_model.h5 tersedia.")
                sys.exit(1)
        
        self.model_path = model_path
        self.is_keras = False
        self.sess = None
        
        # Check if model is Keras (.keras or .h5) or TF1 checkpoint
        if os.path.isfile(model_path) and (model_path.endswith('.keras') or model_path.endswith('.h5')):
            self.is_keras = True
            try:
                print(f"📂 Loading Keras model dari: {model_path}")
                self.model = tf.keras.models.load_model(model_path)
                print("✅ Model Keras berhasil diload!")
            except Exception as e:
                print(f"❌ Gagal loading model Keras: {str(e)}")
                sys.exit(1)
        else:
            # TF1 checkpoint loader
            try:
                tf.compat.v1.disable_eager_execution()
                self.sess = tf.compat.v1.Session()
                meta_path = os.path.join(model_path, 'model_eval.meta')
                ckpt_path = os.path.join(model_path, 'model')
                
                print(f"📂 Loading TF1 checkpoint dari: {model_path}")
                saver = tf.compat.v1.train.import_meta_graph(meta_path)
                saver.restore(self.sess, ckpt_path)
                
                self.graph = tf.compat.v1.get_default_graph()
                self.input_tensor = self.graph.get_tensor_by_name("image:0")
                self.logits_tensor = self.graph.get_tensor_by_name("resnet_model/final_dense:0")
                print("✅ Model TF1 berhasil diload!")
            except Exception as e:
                print(f"❌ Gagal loading model TF1: {str(e)}")
                sys.exit(1)
    
    def preprocess_image(self, image_path):
        """
        Preprocess gambar X-ray sesuai dengan standar TB-Net
        
        Args:
            image_path: Path ke gambar X-ray
            
        Returns:
            numpy array: Gambar yang sudah dipreprocess
        """
        print(f"🔄 Preprocessing gambar: {os.path.basename(image_path)}")
        
        try:
            # Baca gambar
            image_color = cv2.imread(image_path, 1)
            if image_color is None:
                raise ValueError(f"Gagal membaca gambar: {image_path}")
            
            if self.is_keras:
                # Model Keras dilatih menggunakan format RGB standar berukuran (224, 224) dengan normalisasi [0, 1]
                image_rgb = cv2.cvtColor(image_color, cv2.COLOR_BGR2RGB)
                image_resized = cv2.resize(image_rgb, (224, 224), interpolation=cv2.INTER_LANCZOS4)
                image_normalized = image_resized.astype(np.float32) / 255.0
                return image_normalized
            
            # Preprocessing khusus DarwinAI TF1 checkpoint:
            # Ambil channel Blue saja (karena X-ray grayscale)
            b, g, r = cv2.split(image_color)
            height, width = b.shape
            image = b.copy()
            
            # Fungsi untuk deteksi padding/border hitam
            def is_padding(row):
                """Deteksi jika row/column adalah padding hitam"""
                for i in range(6):
                    if not np.count_nonzero(np.subtract(row, i)):
                        return True
                for i in range(250, 256):
                    if not np.count_nonzero(np.subtract(row, i)):
                        return True
                return False
            
            # Remove black borders - atas ke bawah
            min_y = 0
            for row in image:
                if is_padding(row):
                    min_y += 1
                else:
                    break
            
            # Remove black borders - bawah ke atas
            max_y = height - 1
            for row_index in range(height - 1, 0, -1):
                if is_padding(image[row_index]):
                    max_y -= 1
                else:
                    break
            
            # Remove black borders - kiri ke kanan
            min_x = 0
            image_transposed = image.T
            for col in image_transposed:
                if is_padding(col):
                    min_x += 1
                else:
                    break
            
            # Remove black borders - kanan ke kiri
            max_x = width - 1
            for col_index in range(width - 1, 0, -1):
                if is_padding(image_transposed[col_index]):
                    max_x -= 1
                else:
                    break
            
            # Crop image to remove borders if valid crop
            if max_y > min_y and max_x > min_x:
                image = image[min_y:max_y, min_x:max_x]
            
            # Convert to 3 channels
            image = cv2.merge([image, image, image])
            
            # Resize to 224x224
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            
            # Normalize to [0, 1]
            min_value = np.min(image)
            max_value = np.max(image)
            if max_value > min_value:
                image = np.divide(np.subtract(image, min_value), max_value - min_value)
            else:
                image = np.zeros_like(image, dtype=np.float32)
            
            # Additional crop (5% from each side, 25% from bottom)
            image = image[11:168, 11:202]
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LANCZOS4)
            
            # Create masks for corners (to hide metadata)
            avg_train_intensity = (0.531459229, 0.531459229, 0.531459229)
            avg_mask = np.zeros((224, 224, 3), dtype=np.float32)
            black_mask = np.zeros((224, 224, 3), dtype=np.float32)
            
            for r_idx in range(224):
                for c_idx in range(224):
                    if r_idx < 45 and (c_idx < 45 or c_idx > 178):
                        avg_mask[r_idx][c_idx] = avg_train_intensity
                        black_mask[r_idx][c_idx] = (0, 0, 0)
                    else:
                        black_mask[r_idx][c_idx] = (1, 1, 1)
            
            # Apply black mask
            image = np.multiply(image, black_mask)
            
            # Renormalize
            min_value = np.min(image)
            max_value = np.max(image)
            if max_value > min_value:
                image = np.divide(np.subtract(image, min_value), max_value - min_value)
            
            # Add average intensity to corners
            image = cv2.addWeighted(image.astype(np.float32), 1, avg_mask, 1, 0)
            
            return image
            
        except Exception as e:
            print(f"❌ Error preprocessing: {str(e)}")
            return None
    
    def predict_single_image(self, image_path):
        """
        Prediksi satu gambar X-ray
        
        Args:
            image_path: Path ke gambar
            
        Returns:
            dict: Hasil prediksi dengan detail
        """
        print(f"\n📊 Analisis gambar: {os.path.basename(image_path)}")
        
        # Preprocess image
        processed_img = self.preprocess_image(image_path)
        if processed_img is None:
            return None
        
        # Run inference
        try:
            if self.is_keras:
                # Keras model inference
                input_arr = np.expand_dims(processed_img, axis=0)
                preds = self.model.predict(input_arr, verbose=0)[0]
                
                # Check binary or multi-class
                if preds.shape[-1] == 1 or np.ndim(preds) == 0 or len(preds) == 1:
                    tb_prob = float(preds[0]) if hasattr(preds, '__getitem__') else float(preds)
                    normal_prob = float(1.0 - tb_prob)
                    softmax = [normal_prob, tb_prob]
                    pred_class = 1 if tb_prob >= 0.5 else 0
                    confidence = tb_prob if pred_class == 1 else normal_prob
                else:
                    exp_logits = np.exp(preds - np.max(preds))
                    softmax = (exp_logits / np.sum(exp_logits)).tolist()
                    pred_class = int(np.argmax(softmax))
                    confidence = float(softmax[pred_class])
            else:
                # TF1 Session inference
                logits = self.sess.run(self.logits_tensor, 
                                      feed_dict={self.input_tensor: [processed_img]})[0]
                exp_logits = np.exp(logits - np.max(logits))
                softmax = (exp_logits / np.sum(exp_logits)).tolist()
                pred_class = int(np.argmax(softmax))
                confidence = float(softmax[pred_class])
            
            # Interpretation
            if pred_class == 0:  # Normal
                diagnosis = "NORMAL"
                recommendation = "Tidak terdeteksi tanda-tanda Tuberculosis. Tetap jaga kesehatan paru-paru dengan pola hidup sehat."
                color = "🟢"
                risk_level = "RENDAH"
            else:  # Tuberculosis
                diagnosis = "TUBERCULOSIS"
                recommendation = "TERDETEKSI kemungkinan Tuberculosis. Segera konsultasi ke dokter spesialis paru untuk pemeriksaan lebih lanjut."
                color = "🔴"
                risk_level = "TINGGI"
            
            result = {
                'filename': os.path.basename(image_path),
                'diagnosis': diagnosis,
                'confidence': float(confidence),
                'confidence_percent': f"{confidence*100:.2f}%",
                'normal_prob': float(softmax[0]),
                'tb_prob': float(softmax[1]),
                'recommendation': recommendation,
                'color': color,
                'risk_level': risk_level,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'processed_image': processed_img
            }
            
            # Print result
            print(f"\n{'='*50}")
            print(f"{color} HASIL DETEKSI:")
            print(f"{'='*50}")
            print(f"📁 File: {result['filename']}")
            print(f"🏥 Diagnosis: {result['diagnosis']}")
            print(f"📈 Confidence: {result['confidence_percent']}")
            print(f"📊 Probabilitas Normal: {softmax[0]:.4f}")
            print(f"📊 Probabilitas TB: {softmax[1]:.4f}")
            print(f"⚠️  Level Risiko: {result['risk_level']}")
            print(f"\n💡 Rekomendasi:")
            print(f"   {result['recommendation']}")
            print(f"{'='*50}")
            
            return result
            
        except Exception as e:
            print(f"❌ Error selama prediksi: {str(e)}")
            return None
    
    def predict_batch(self, input_folder, output_folder='results'):
        """
        Prediksi semua gambar dalam folder
        
        Args:
            input_folder: Folder berisi gambar X-ray
            output_folder: Folder untuk menyimpan hasil
            
        Returns:
            DataFrame: Hasil semua prediksi
        """
        print(f"\n📂 Memproses semua gambar di: {input_folder}")
        
        # Create output folder
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all image files
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
        image_files = []
        
        for root, dirs, files in os.walk(input_folder):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            print("❌ Tidak ditemukan gambar di folder tersebut")
            return None
        
        print(f"📷 Ditemukan {len(image_files)} gambar")
        
        # Process each image
        results = []
        for i, img_path in enumerate(image_files, 1):
            print(f"\n[{i}/{len(image_files)}] Memproses: {os.path.basename(img_path)}")
            
            result = self.predict_single_image(img_path)
            if result:
                results.append(result)
        
        # Save to CSV
        if results:
            df = pd.DataFrame(results)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            csv_path = os.path.join(output_folder, f"hasil_deteksi_{timestamp}.csv")
            
            # Save simplified CSV
            df_simple = df[['filename', 'diagnosis', 'confidence_percent', 'risk_level', 'timestamp']]
            df_simple.to_csv(csv_path, index=False, encoding='utf-8-sig')
            
            # Save detailed CSV (exclude heavy processed_image array from CSV)
            detailed_path = os.path.join(output_folder, f"hasil_deteksi_detail_{timestamp}.csv")
            df_detail = df[['filename', 'diagnosis', 'confidence', 'confidence_percent', 'normal_prob', 'tb_prob', 'risk_level', 'recommendation', 'timestamp']]
            df_detail.to_csv(detailed_path, index=False, encoding='utf-8-sig')
            
            print(f"\n✅ Hasil disimpan di:")
            print(f"   📄 Ringkasan: {csv_path}")
            print(f"   📄 Detail: {detailed_path}")
            
            # Print summary
            self.print_summary(df)
            
            return df
        else:
            print("❌ Tidak ada hasil yang diperoleh")
            return None
    
    def print_summary(self, df):
        """Print summary statistics"""
        print(f"\n{'='*50}")
        print("📊 RINGKASAN HASIL DETEKSI")
        print(f"{'='*50}")
        
        total = len(df)
        normal_count = len(df[df['diagnosis'] == 'NORMAL'])
        tb_count = len(df[df['diagnosis'] == 'TUBERCULOSIS'])
        
        print(f"📈 Total Gambar: {total}")
        print(f"🟢 Normal: {normal_count} ({normal_count/total*100:.1f}%)")
        print(f"🔴 Tuberculosis: {tb_count} ({tb_count/total*100:.1f}%)")
        
        if tb_count > 0:
            avg_tb_confidence = df[df['diagnosis'] == 'TUBERCULOSIS']['confidence'].mean()
            print(f"📊 Rata-rata confidence TB: {avg_tb_confidence*100:.1f}%")
        
        print(f"{'='*50}")
    
    def visualize_result(self, result, save_path=None, show_plot=False):
        """
        Visualisasi hasil prediksi
        
        Args:
            result: Hasil prediksi dari predict_single_image()
            save_path: Path untuk menyimpan gambar (optional)
            show_plot: Apakah menampilkan jendela grafik interaktif
        """
        if result is None:
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot probabilities
        classes = ['Normal', 'Tuberculosis']
        probs = [result['normal_prob'], result['tb_prob']]
        colors = ['green', 'red']
        
        axes[0].bar(classes, probs, color=colors, alpha=0.7)
        axes[0].set_ylabel('Probabilitas')
        axes[0].set_title('Probabilitas Deteksi')
        axes[0].set_ylim([0, 1])
        
        # Add value labels
        for i, v in enumerate(probs):
            axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center')
        
        # Plot processed image
        axes[1].imshow(result['processed_image'])
        axes[1].set_title(f'Gambar Hasil Preprocessing\nDiagnosis: {result["diagnosis"]}')
        axes[1].axis('off')
        
        plt.suptitle(f'Hasil Deteksi: {result["filename"]}\nConfidence: {result["confidence_percent"]}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Visualisasi disimpan di: {save_path}")
        
        if show_plot:
            try:
                plt.show()
            except Exception:
                pass
        plt.close(fig)
    
    def close(self):
        """Close TensorFlow session if opened"""
        if not self.is_keras and self.sess is not None:
            self.sess.close()
        print("👋 Sesi deteksi ditutup")


def main():
    """Fungsi utama"""
    default_output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    parser = argparse.ArgumentParser(description='Sistem Deteksi Tuberculosis dari X-Ray')
    parser.add_argument('--input', type=str, help='Path ke gambar atau folder')
    parser.add_argument('--output', type=str, default=default_output_dir, help='Folder output')
    parser.add_argument('--model', type=str, default=None, help='Path ke model (.keras, .h5, atau folder TF ckpt)')
    parser.add_argument('--visualize', action='store_true', help='Tampilkan visualisasi')
    
    args = parser.parse_args()
    
    # Create system
    detector = TBDetectionSystem(model_path=args.model)
    
    try:
        if args.input:
            if os.path.isfile(args.input):
                # Single image
                result = detector.predict_single_image(args.input)
                
                if args.visualize and result:
                    # Create visualization
                    vis_path = os.path.join(args.output, f"visualisasi_{os.path.splitext(os.path.basename(args.input))[0]}.png")
                    os.makedirs(args.output, exist_ok=True)
                    detector.visualize_result(result, save_path=vis_path)
                
                # Save single result
                if result:
                    df = pd.DataFrame([{k: v for k, v in result.items() if k != 'processed_image'}])
                    csv_path = os.path.join(args.output, f"hasil_{os.path.splitext(os.path.basename(args.input))[0]}.csv")
                    os.makedirs(args.output, exist_ok=True)
                    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
                    print(f"✅ Hasil disimpan di: {csv_path}")
            
            elif os.path.isdir(args.input):
                # Batch processing
                detector.predict_batch(args.input, args.output)
            
            else:
                print(f"❌ Path tidak valid: {args.input}")
        
        else:
            # Interactive mode
            print("\n" + "="*60)
            print("🏥 SISTEM DETEKSI TUBERCULOSIS (TB CHECK)")
            print("="*60)
            print("\nPilih mode:")
            print("1. Deteksi satu gambar")
            print("2. Deteksi seluruh folder")
            print("3. Keluar")
            
            choice = input("\nPilihan Anda (1/2/3): ").strip()
            
            if choice == '1':
                img_path = input("Masukkan path ke gambar X-ray: ").strip()
                if os.path.exists(img_path):
                    result = detector.predict_single_image(img_path)
                    if result:
                        viz = input("Tampilkan visualisasi? (y/n): ").lower()
                        if viz == 'y':
                            detector.visualize_result(result, show_plot=True)
                else:
                    print("❌ File tidak ditemukan")
            
            elif choice == '2':
                folder_path = input("Masukkan path ke folder: ").strip()
                if os.path.exists(folder_path):
                    detector.predict_batch(folder_path)
                else:
                    print("❌ Folder tidak ditemukan")
            
            elif choice == '3':
                print("Terima kasih telah menggunakan sistem deteksi TB!")
            
            else:
                print("❌ Pilihan tidak valid")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Program dihentikan oleh pengguna")
    
    except Exception as e:
        print(f"\n❌ Terjadi error: {str(e)}")
    
    finally:
        detector.close()


if __name__ == "__main__":
    main()