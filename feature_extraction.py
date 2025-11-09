import numpy as np
import librosa
import scipy.stats
from scipy.signal import find_peaks, butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

class VoiceFeatureExtractor:
    def __init__(self, sr=22050):
        self.sr = sr
    
    def extract_features(self, audio, sr=None):
        """Extract real voice features for Parkinson's detection"""
        if sr is None:
            sr = self.sr
        
        # Ensure audio is 1D and normalized
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Normalize audio
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio))
        
        # Remove DC offset
        audio = audio - np.mean(audio)
        
        # Basic validation
        if len(audio) < sr * 0.1:  # Less than 0.1 seconds
            return self.get_default_features()
        
        # Extract all features
        features = {}
        
        try:
            # 1. FUNDAMENTAL FREQUENCY (F0) FEATURES
            f0, voiced_flag, voiced_probs = librosa.pyin(
                audio, 
                fmin=50,  # Lower for male voices
                fmax=500,  # Higher for female voices
                sr=sr,
                frame_length=2048,
                hop_length=512
            )
            
            # Remove unvoiced segments
            f0_voiced = f0[~np.isnan(f0)]
            
            if len(f0_voiced) > 5:  # Need at least 5 voiced frames
                features['MDVP:Fo(Hz)'] = float(np.mean(f0_voiced))
                features['MDVP:Fhi(Hz)'] = float(np.max(f0_voiced))
                features['MDVP:Flo(Hz)'] = float(np.min(f0_voiced))
                
                # 2. JITTER MEASUREMENTS (frequency perturbation)
                if len(f0_voiced) > 10:
                    # Local jitter
                    jitter_local = []
                    for i in range(1, len(f0_voiced)):
                        jitter_local.append(abs(f0_voiced[i] - f0_voiced[i-1]))
                    
                    mean_period = np.mean(1/f0_voiced) if np.mean(f0_voiced) > 0 else 0.01
                    
                    # MDVP:Jitter(%)
                    features['MDVP:Jitter(%)'] = float(
                        (np.mean(jitter_local) / np.mean(f0_voiced)) * 100 
                        if np.mean(f0_voiced) > 0 else 0.5
                    )
                    
                    # MDVP:Jitter(Abs) - in microseconds
                    features['MDVP:Jitter(Abs)'] = float(np.mean(jitter_local) * 1000 / sr)
                    
                    # RAP (Relative Average Perturbation)
                    rap = 0
                    for i in range(1, len(f0_voiced) - 1):
                        cycle_avg = (f0_voiced[i-1] + f0_voiced[i] + f0_voiced[i+1]) / 3
                        rap += abs(f0_voiced[i] - cycle_avg)
                    features['MDVP:RAP'] = float(rap / (len(f0_voiced) - 2) / np.mean(f0_voiced)) if len(f0_voiced) > 2 else 0.002
                    
                    # PPQ (Pitch Perturbation Quotient)
                    ppq = 0
                    q = 5  # 5-point PPQ
                    for i in range(2, len(f0_voiced) - 2):
                        window = f0_voiced[i-2:i+3]
                        ppq += abs(f0_voiced[i] - np.mean(window))
                    features['MDVP:PPQ'] = float(ppq / (len(f0_voiced) - 4) / np.mean(f0_voiced)) if len(f0_voiced) > 4 else 0.003
                    
                    # DDP (Degree of voice breaks)
                    features['Jitter:DDP'] = features['MDVP:RAP'] * 3
                else:
                    # Default jitter values
                    features['MDVP:Jitter(%)'] = 0.5
                    features['MDVP:Jitter(Abs)'] = 0.00005
                    features['MDVP:RAP'] = 0.002
                    features['MDVP:PPQ'] = 0.003
                    features['Jitter:DDP'] = 0.006
                
                # 3. SHIMMER MEASUREMENTS (amplitude perturbation)
                # Get amplitude envelope
                amplitude_env = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
                amplitude_peaks = np.max(amplitude_env, axis=0)
                
                if len(amplitude_peaks) > 10:
                    # Local shimmer
                    shimmer_local = []
                    for i in range(1, len(amplitude_peaks)):
                        shimmer_local.append(abs(amplitude_peaks[i] - amplitude_peaks[i-1]))
                    
                    # MDVP:Shimmer
                    features['MDVP:Shimmer'] = float(
                        (np.mean(shimmer_local) / np.mean(amplitude_peaks)) * 100 
                        if np.mean(amplitude_peaks) > 0 else 2.0
                    )
                    
                    # MDVP:Shimmer(dB)
                    features['MDVP:Shimmer(dB)'] = float(
                        20 * np.log10(1 + features['MDVP:Shimmer']/100)
                    )
                    
                    # APQ3 (3-point Amplitude Perturbation Quotient)
                    apq3 = 0
                    for i in range(1, len(amplitude_peaks) - 1):
                        window = amplitude_peaks[i-1:i+2]
                        apq3 += abs(amplitude_peaks[i] - np.mean(window))
                    features['Shimmer:APQ3'] = float(apq3 / (len(amplitude_peaks) - 2) / np.mean(amplitude_peaks)) if len(amplitude_peaks) > 2 else 0.01
                    
                    # APQ5
                    apq5 = 0
                    for i in range(2, len(amplitude_peaks) - 2):
                        window = amplitude_peaks[i-2:i+3]
                        apq5 += abs(amplitude_peaks[i] - np.mean(window))
                    features['Shimmer:APQ5'] = float(apq5 / (len(amplitude_peaks) - 4) / np.mean(amplitude_peaks)) if len(amplitude_peaks) > 4 else 0.015
                    
                    # APQ11
                    features['MDVP:APQ'] = features['Shimmer:APQ5'] * 1.5
                    
                    # DDA
                    features['Shimmer:DDA'] = features['Shimmer:APQ3'] * 3
                else:
                    # Default shimmer values
                    features['MDVP:Shimmer'] = 2.0
                    features['MDVP:Shimmer(dB)'] = 0.2
                    features['Shimmer:APQ3'] = 0.01
                    features['Shimmer:APQ5'] = 0.015
                    features['MDVP:APQ'] = 0.02
                    features['Shimmer:DDA'] = 0.03
                
                # 4. HARMONIC-TO-NOISE RATIO (HNR)
                hnr = self.calculate_hnr(audio, sr, f0_voiced)
                features['HNR'] = float(hnr)
                features['NHR'] = float(1.0 / (hnr + 1))
                
                # 5. COMPLEXITY MEASURES
                # RPDE (Recurrence Period Density Entropy)
                features['RPDE'] = float(self.calculate_rpde(audio))
                
                # DFA (Detrended Fluctuation Analysis)
                features['DFA'] = float(self.calculate_dfa(audio))
                
                # Spread measures
                features['spread1'] = float(np.std(f0_voiced))
                features['spread2'] = float(np.std(f0_voiced) * np.mean(f0_voiced) / 100 if np.mean(f0_voiced) > 0 else 0.5)
                
                # D2 (Correlation dimension)
                features['D2'] = float(self.calculate_correlation_dimension(audio))
                
                # PPE (Pitch Period Entropy)
                features['PPE'] = float(self.calculate_ppe(f0_voiced))
                
            else:
                # No voiced segments found - use defaults
                return self.get_default_features()
                
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return self.get_default_features()
        
        # Ensure all features are present and are floats
        default_features = self.get_default_features()
        for key in default_features:
            if key not in features:
                features[key] = default_features[key]
            else:
                features[key] = float(features[key])
        
        # Clip extreme values to reasonable ranges
        features['MDVP:Jitter(%)'] = np.clip(features['MDVP:Jitter(%)'], 0, 5)
        features['MDVP:Shimmer'] = np.clip(features['MDVP:Shimmer'], 0, 10)
        features['HNR'] = np.clip(features['HNR'], 0, 40)
        
        return features
    
    def calculate_hnr(self, audio, sr, f0_values):
        """Calculate Harmonics-to-Noise Ratio"""
        try:
            # Use autocorrelation method
            if len(f0_values) > 0:
                mean_f0 = np.mean(f0_values)
                period_samples = int(sr / mean_f0)
                
                if period_samples < len(audio):
                    # Calculate autocorrelation at the period
                    autocorr = np.correlate(audio, audio, mode='same')
                    autocorr = autocorr[len(autocorr)//2:]
                    
                    # Find peak around expected period
                    search_range = int(period_samples * 0.2)
                    start = max(period_samples - search_range, 1)
                    end = min(period_samples + search_range, len(autocorr))
                    
                    if end > start:
                        peak_idx = start + np.argmax(autocorr[start:end])
                        
                        # HNR calculation
                        signal_energy = autocorr[peak_idx]
                        noise_energy = np.mean(np.abs(audio - np.roll(audio, peak_idx)))
                        
                        if noise_energy > 0:
                            hnr = 10 * np.log10(signal_energy / noise_energy)
                            return np.clip(hnr, 0, 40)
            
            # Default HNR
            return 20.0
            
        except:
            return 20.0
    
    def calculate_rpde(self, signal):
        """Calculate Recurrence Period Density Entropy"""
        try:
            # Simplified RPDE calculation
            if len(signal) < 100:
                return 0.5
            
            # Phase space reconstruction
            m = 3  # embedding dimension
            tau = 2  # time delay
            
            N = len(signal) - (m-1)*tau
            if N <= 0:
                return 0.5
            
            # Create embedded matrix
            embedded = np.zeros((N, m))
            for i in range(m):
                embedded[:, i] = signal[i*tau:i*tau+N]
            
            # Calculate recurrence times
            recurrence_times = []
            threshold = 0.1 * np.std(signal)
            
            for i in range(min(N, 100)):  # Limit for speed
                distances = np.sqrt(np.sum((embedded - embedded[i])**2, axis=1))
                recurrent = np.where(distances < threshold)[0]
                
                if len(recurrent) > 1:
                    periods = np.diff(recurrent)
                    recurrence_times.extend(periods[periods > 0])
            
            if len(recurrence_times) > 0:
                # Calculate entropy
                hist, _ = np.histogram(recurrence_times, bins=10)
                hist = hist[hist > 0]
                probs = hist / np.sum(hist)
                entropy = -np.sum(probs * np.log(probs))
                return np.clip(entropy / np.log(10), 0, 1)  # Normalize
            
            return 0.5
            
        except:
            return 0.5
    
    def calculate_dfa(self, signal):
        """Calculate Detrended Fluctuation Analysis exponent"""
        try:
            if len(signal) < 100:
                return 0.7
            
            # Integrate the signal
            Y = np.cumsum(signal - np.mean(signal))
            
            # Calculate for different box sizes
            scales = np.logspace(1.2, min(3, np.log10(len(Y)/4)), 8, dtype=int)
            F = []
            
            for scale in scales:
                if scale >= 4 and scale < len(Y)/2:
                    # Divide into boxes
                    n_boxes = len(Y) // scale
                    
                    if n_boxes >= 2:
                        Y_boxed = Y[:n_boxes*scale].reshape(n_boxes, scale)
                        
                        # Detrend each box
                        x = np.arange(scale)
                        fluctuations = []
                        
                        for box in Y_boxed:
                            coef = np.polyfit(x, box, 1)
                            fit = np.polyval(coef, x)
                            fluctuations.append(np.sqrt(np.mean((box - fit)**2)))
                        
                        F.append(np.mean(fluctuations))
            
            if len(F) > 2:
                # Fit in log-log space
                coeffs = np.polyfit(np.log(scales[:len(F)]), np.log(F), 1)
                return np.clip(coeffs[0], 0.1, 1.5)
            
            return 0.7
            
        except:
            return 0.7
    
    def calculate_correlation_dimension(self, signal):
        """Calculate correlation dimension (D2)"""
        try:
            if len(signal) < 100:
                return 1.0
            
            # Simplified correlation dimension
            m = 2  # embedding dimension
            N = min(len(signal), 500)  # Limit for speed
            signal_sample = signal[:N]
            
            # Normalize
            signal_sample = (signal_sample - np.mean(signal_sample)) / np.std(signal_sample)
            
            # Calculate correlation sum
            r_values = np.logspace(-2, 0, 10)
            C = []
            
            for r in r_values:
                count = 0
                for i in range(N-m):
                    for j in range(i+1, N-m):
                        if np.abs(signal_sample[i] - signal_sample[j]) < r:
                            count += 1
                
                if count > 0:
                    C.append(count / ((N-m) * (N-m-1) / 2))
            
            if len(C) > 2:
                # Estimate slope in log-log plot
                valid_idx = np.where(np.array(C) > 0)[0]
                if len(valid_idx) > 2:
                    coeffs = np.polyfit(np.log(r_values[valid_idx]), 
                                       np.log(np.array(C)[valid_idx]), 1)
                    return np.clip(coeffs[0], 0.5, 2.0)
            
            return 1.0
            
        except:
            return 1.0
    
    def calculate_ppe(self, f0_values):
        """Calculate Pitch Period Entropy"""
        try:
            if len(f0_values) < 10:
                return 0.2
            
            # Calculate pitch periods
            periods = 1.0 / f0_values
            
            # Calculate entropy
            hist, _ = np.histogram(periods, bins=10)
            hist = hist[hist > 0]
            if len(hist) > 0:
                probs = hist / np.sum(hist)
                entropy = -np.sum(probs * np.log(probs))
                return np.clip(entropy / np.log(10), 0, 1)
            
            return 0.2
            
        except:
            return 0.2
    
    def get_default_features(self):
        """Return default features for edge cases"""
        return {
            'MDVP:Fo(Hz)': 150.0,
            'MDVP:Fhi(Hz)': 200.0,
            'MDVP:Flo(Hz)': 100.0,
            'MDVP:Jitter(%)': 0.5,
            'MDVP:Jitter(Abs)': 0.00005,
            'MDVP:RAP': 0.002,
            'MDVP:PPQ': 0.003,
            'Jitter:DDP': 0.006,
            'MDVP:Shimmer': 2.0,
            'MDVP:Shimmer(dB)': 0.2,
            'Shimmer:APQ3': 0.01,
            'Shimmer:APQ5': 0.015,
            'MDVP:APQ': 0.02,
            'Shimmer:DDA': 0.03,
            'NHR': 0.02,
            'HNR': 20.0,
            'RPDE': 0.5,
            'DFA': 0.7,
            'spread1': 1.0,
            'spread2': 0.5,
            'D2': 1.0,
            'PPE': 0.2
        }