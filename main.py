import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.fft import fft, fftfreq
import pandas as pd
from dataclasses import dataclass
import json

@dataclass
class SimulationConfig:
    """Simülasyon konfigürasyonu"""
    # Giriş parametreleri
    V_in: float = 12.0              # Giriş gerilimi (V)
    V_boost_target: float = 340.0   # Hedef boost gerilimi (V)
    V_rms_target: float = 220.0     # Hedef RMS çıkış gerilimi (V)
    
    # Frekans parametreleri
    f_grid: float = 50.0            # Şebeke frekansı (Hz)
    f_pwm: float = 20000.0          # PWM frekansı (Hz)
    f_sample: float = 300000.0      # Örnekleme frekansı (Hz)
    
    # Boost devre elemanları
    L_boost: float = 100e-6         # Boost indüktör (H)
    C_boost: float = 470e-6         # Boost kondansatör (F)
    R_load: float = 50.0            # Yük direnci (Ohm)
    ESR_L: float = 0.05             # İndüktör seri direnci (Ohm)
    ESR_C: float = 0.01             # Kondansatör seri direnci (Ohm)
    
    # Inverter LC filtresi
    L_filter: float = 2e-3          # Filtre indüktörü (H)
    C_filter: float = 10e-6         # Filtre kondansatörü (F)
    
    # Kontrol parametreleri
    boost_ki: float = 50.0          # Boost PI kontrol I kazancı
    boost_kp: float = 0.005         # Boost PI kontrol P kazancı
    deadtime: float = 1e-6          # H-bridge ölü zaman (s)
    
    # Simülasyon parametreleri
    t_sim: float = 0.1              # Simülasyon süresi (s)
    save_decimation: int = 10       # Kayıt decimation faktörü

class BoostInverterSimulation:
    def __init__(self, config: SimulationConfig = None):
        self.config = config or SimulationConfig()
        
        # Simülasyon zaman parametreleri
        self.dt = 1.0 / self.config.f_sample
        self.n_samples = int(self.config.t_sim * self.config.f_sample)
        
        # Durum değişkenleri
        self.i_L_boost = 0.0
        self.v_C_boost = 0.0
        self.i_L_filter = 0.0
        self.v_C_filter = 0.0
        
        # Kontrol değişkenleri
        self.boost_error_integral = 0.0
        self.boost_duty = 0.7
        self.modulation_index = 0.9
        
        # Veri kayıt
        self.data = {
            'time': [],
            'v_boost': [],
            'v_output': [],
            'i_boost': [],
            'i_output': [],
            'duty_cycle': [],
            'pwm_signal': [],
            'power_in': [],
            'power_out': []
        }
        
        # İstatistikler
        self.stats = {
            'avg_v_boost': 0,
            'avg_power_in': 0,
            'avg_power_out': 0,
            'efficiency': 0,
            'thd': 0,
            'v_rms_output': 0
        }
        
    def spwm_modulation(self, t):
        """Gelişmiş SPWM modülasyonu"""
        # Sinüzoidal referans (modülasyon indeksi ile)
        v_ref = self.modulation_index * np.sin(2 * np.pi * self.config.f_grid * t)
        
        # Üçgen taşıyıcı
        phase = (t * self.config.f_pwm) % 1.0
        carrier = 2.0 * phase - 1.0
        
        # SPWM karşılaştırma
        if v_ref > carrier:
            pwm_state = 1.0
        elif v_ref < -carrier:
            pwm_state = -1.0
        else:
            pwm_state = 0.0  # Ölü zaman
            
        return pwm_state
    
    def pi_controller(self, error, ki, kp):
        """PI kontrolör"""
        self.boost_error_integral += error * self.dt
        # Anti-windup
        self.boost_error_integral = np.clip(self.boost_error_integral, -100, 100)
        
        output = kp * error + ki * self.boost_error_integral
        return np.clip(output, 0.1, 0.90)
    
    def boost_converter_step(self):
        """Gelişmiş boost konvertör modeli (PI kontrol ile)"""
        # PI kontrolör ile duty cycle hesapla
        error = self.config.V_boost_target - self.v_C_boost
        self.boost_duty = self.pi_controller(error, self.config.boost_ki, self.config.boost_kp)
        
        # PWM sinyali üret (yüksek frekanslı switching)
        pwm_phase = (self.current_time * self.config.f_pwm) % 1.0
        switch_on = pwm_phase < self.boost_duty
        
        if switch_on:
            # Anahtar kapalı: indüktör şarj oluyor
            v_L = self.config.V_in - self.i_L_boost * self.config.ESR_L
            di_L = v_L / self.config.L_boost
            # Kondansatör sadece yükü besliyor
            i_load = self.v_C_boost / self.config.R_load
            dv_C = -i_load / self.config.C_boost
        else:
            # Anahtar açık: indüktör deşarj, kondansatör şarj
            v_L = self.config.V_in - self.v_C_boost - self.i_L_boost * self.config.ESR_L
            di_L = v_L / self.config.L_boost
            # Kondansatör hem indüktörden şarj oluyor hem yükü besliyor
            i_load = self.v_C_boost / self.config.R_load
            dv_C = (self.i_L_boost - i_load) / self.config.C_boost
        
        # Durum değişkenlerini güncelle
        self.i_L_boost += di_L * self.dt
        self.v_C_boost += dv_C * self.dt
        
        # Fiziksel limitler
        self.i_L_boost = max(0, self.i_L_boost)
        self.v_C_boost = max(0, min(400, self.v_C_boost))  # Max 400V güvenlik limiti
        
    def inverter_step(self, pwm_signal):
        """H-bridge inverter ve LC filtre"""
        # H-bridge çıkışı (ölü zaman dahil)
        if abs(pwm_signal) < 0.1:  # Ölü zaman bölgesi
            v_bridge = 0.0
        else:
            v_bridge = pwm_signal * self.v_C_boost
        
        # LC filtre diferansiyel denklemleri
        di_L = (v_bridge - self.v_C_filter - self.i_L_filter * self.config.ESR_L) / self.config.L_filter
        dv_C = (self.i_L_filter - self.v_C_filter / self.config.R_load) / self.config.C_filter
        
        self.i_L_filter += di_L * self.dt
        self.v_C_filter += dv_C * self.dt
        
        return v_bridge
        
    def calculate_power(self):
        """Anlık güç hesaplama"""
        p_in = self.config.V_in * self.i_L_boost
        p_out = self.v_C_filter * (self.v_C_filter / self.config.R_load)
        return p_in, p_out
        
    def run_simulation(self):
        """Ana simülasyon döngüsü"""
        print("=" * 60)
        print("BOOST INVERTER SIMULASYONU BASLATILIYOR")
        print("=" * 60)
        print(f"Giris: {self.config.V_in}V DC -> Boost: {self.config.V_boost_target}V DC")
        print(f"Cikis: {self.config.V_rms_target}V RMS @ {self.config.f_grid}Hz")
        print(f"PWM Frekansi: {self.config.f_pwm/1000:.0f} kHz")
        print(f"Ornekleme: {self.config.f_sample/1000:.0f} kHz")
        print(f"Toplam ornek: {self.n_samples:,}")
        print("=" * 60)
        
        # Boost başlangıç şartlandırma (yavaşça yükselsin)
        self.v_C_boost = 50.0  # Düşük başlangıç
        
        progress_points = [int(self.n_samples * i / 20) for i in range(21)]
        
        for i in range(self.n_samples):
            self.current_time = i * self.dt
            
            # SPWM sinyali
            pwm = self.spwm_modulation(self.current_time)
            
            # Boost konvertör
            self.boost_converter_step()
            
            # Inverter
            v_bridge = self.inverter_step(pwm)
            
            # Güç hesaplama
            p_in, p_out = self.calculate_power()
            
            # Veri kaydet
            if i % self.config.save_decimation == 0:
                self.data['time'].append(self.current_time)
                self.data['v_boost'].append(self.v_C_boost)
                self.data['v_output'].append(self.v_C_filter)
                self.data['i_boost'].append(self.i_L_boost)
                self.data['i_output'].append(self.i_L_filter)
                self.data['duty_cycle'].append(self.boost_duty)
                self.data['pwm_signal'].append(pwm)
                self.data['power_in'].append(p_in)
                self.data['power_out'].append(p_out)
            
            # İlerleme
            if i in progress_points:
                progress = (i / self.n_samples) * 100
                bar_len = int(progress / 5)
                bar = "#" * bar_len + "-" * (20 - bar_len)
                print(f"\r[{bar}] {progress:.0f}% - VBoost: {self.v_C_boost:.1f}V", end='')
        
        print("\n" + "=" * 60)
        self.calculate_statistics()
        print("Simulasyon tamamlandi!")
        print("=" * 60)
        
    def calculate_statistics(self):
        """İstatistik hesaplamaları"""
        # RMS hesaplama (son periyot)
        period_samples = int(len(self.data['time']) / (self.config.f_grid * self.config.t_sim))
        last_period = self.data['v_output'][-period_samples:]
        
        self.stats['v_rms_output'] = np.sqrt(np.mean(np.array(last_period)**2))
        self.stats['avg_v_boost'] = np.mean(self.data['v_boost'][-1000:])
        self.stats['avg_power_in'] = np.mean(self.data['power_in'][-1000:])
        self.stats['avg_power_out'] = np.mean(self.data['power_out'][-1000:])
        
        if self.stats['avg_power_in'] > 0:
            self.stats['efficiency'] = (self.stats['avg_power_out'] / self.stats['avg_power_in']) * 100
        
        # THD hesaplama (FFT ile)
        self.stats['thd'] = self.calculate_thd()
        
    def calculate_thd(self):
        """Total Harmonic Distortion hesaplama"""
        signal = np.array(self.data['v_output'])
        n = len(signal)
        
        # FFT
        yf = fft(signal)
        xf = fftfreq(n, self.dt * self.config.save_decimation)
        
        # Pozitif frekanslar
        pos_mask = xf > 0
        xf_pos = xf[pos_mask]
        yf_pos = np.abs(yf[pos_mask])
        
        # Temel frekans (50 Hz) ve harmonikler
        f_fund_idx = np.argmin(np.abs(xf_pos - self.config.f_grid))
        fundamental = yf_pos[f_fund_idx]
        
        # İlk 10 harmonik
        harmonic_power = 0
        for h in range(2, 11):
            h_freq = h * self.config.f_grid
            h_idx = np.argmin(np.abs(xf_pos - h_freq))
            harmonic_power += yf_pos[h_idx]**2
        
        if fundamental > 0:
            thd = 100 * np.sqrt(harmonic_power) / fundamental
        else:
            thd = 0
            
        return thd
        
    def save_to_csv(self, filename="waveform_advanced.csv"):
        """CSV kaydetme"""
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)
        print(f"\nVeriler '{filename}' dosyasina kaydedildi.")
        return filename
        
    def save_config(self, filename="simulation_config.json"):
        """Konfigürasyonu kaydet"""
        config_dict = {
            'V_in': self.config.V_in,
            'V_boost_target': self.config.V_boost_target,
            'V_rms_target': self.config.V_rms_target,
            'f_grid': self.config.f_grid,
            'f_pwm': self.config.f_pwm,
            'L_boost': self.config.L_boost,
            'C_boost': self.config.C_boost,
            'R_load': self.config.R_load,
            't_sim': self.config.t_sim,
            'statistics': self.stats
        }
        with open(filename, 'w') as f:
            json.dump(config_dict, f, indent=4)
        print(f"Konfigurasyon '{filename}' dosyasina kaydedildi.")
        
    def plot_results(self):
        """Gelişmiş görselleştirme"""
        t = np.array(self.data['time']) * 1000  # ms cinsine çevir
        
        # Filtre uygula
        fc = 500
        b, a = butter(2, fc / (self.config.f_sample / 2), btype='low')
        v_out_filtered = filtfilt(b, a, self.data['v_output'])
        
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
        
        # 1. Boost Gerilimi
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(t, self.data['v_boost'], 'b-', linewidth=1.5, label='Boost Gerilimi')
        ax1.axhline(self.config.V_boost_target, color='r', linestyle='--', 
                   label=f'Hedef: {self.config.V_boost_target}V')
        ax1.set_title('Boost Konvertör Çıkış Gerilimi', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Zaman (ms)')
        ax1.set_ylabel('Gerilim (V)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. İnverter Çıkışı (Tüm)
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(t, v_out_filtered, 'g-', linewidth=2, label='Filtrelenmiş Çıkış')
        peak_target = self.config.V_rms_target * np.sqrt(2)
        ax2.axhline(peak_target, color='r', linestyle='--', alpha=0.5, label=f'Peak: ±{peak_target:.0f}V')
        ax2.axhline(-peak_target, color='r', linestyle='--', alpha=0.5)
        ax2.set_title(f'İnverter Çıkış Gerilimi - RMS: {self.stats["v_rms_output"]:.1f}V', 
                     fontsize=14, fontweight='bold')
        ax2.set_xlabel('Zaman (ms)')
        ax2.set_ylabel('Gerilim (V)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Yakınlaştırma (2 periyot)
        period = 1000.0 / self.config.f_grid  # ms
        zoom_samples = int(2 * period / (t[1] - t[0]))
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(t[:zoom_samples], v_out_filtered[:zoom_samples], 'r-', linewidth=2)
        ax3.set_title('İlk 2 Periyot Yakınlaştırma', fontsize=12, fontweight='bold')
        ax3.set_xlabel('Zaman (ms)')
        ax3.set_ylabel('Gerilim (V)')
        ax3.grid(True, alpha=0.3)
        
        # 4. Güç Analizi
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.plot(t, self.data['power_in'], 'b-', linewidth=1, label='Giriş Gücü', alpha=0.7)
        ax4.plot(t, self.data['power_out'], 'r-', linewidth=1, label='Çıkış Gücü', alpha=0.7)
        ax4.set_title(f'Güç Analizi - Verim: {self.stats["efficiency"]:.1f}%', 
                     fontsize=12, fontweight='bold')
        ax4.set_xlabel('Zaman (ms)')
        ax4.set_ylabel('Güç (W)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Akım Analizi
        ax5 = fig.add_subplot(gs[3, 0])
        ax5.plot(t, self.data['i_boost'], 'b-', linewidth=1, label='Boost Akımı')
        ax5.set_title('Boost İndüktör Akımı', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Zaman (ms)')
        ax5.set_ylabel('Akım (A)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. FFT Analizi
        ax6 = fig.add_subplot(gs[3, 1])
        signal = np.array(self.data['v_output'])
        n = len(signal)
        yf = fft(signal)
        xf = fftfreq(n, self.dt * self.config.save_decimation)
        pos_mask = xf > 0
        xf_pos = xf[pos_mask]
        yf_pos = 2.0/n * np.abs(yf[pos_mask])
        
        ax6.semilogy(xf_pos, yf_pos, 'b-', linewidth=1)
        ax6.axvline(self.config.f_grid, color='r', linestyle='--', label=f'{self.config.f_grid} Hz')
        ax6.set_xlim(0, 500)
        ax6.set_title(f'FFT Spektrumu - THD: {self.stats["thd"]:.2f}%', 
                     fontsize=12, fontweight='bold')
        ax6.set_xlabel('Frekans (Hz)')
        ax6.set_ylabel('Genlik')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        plt.savefig('boost_inverter_advanced.png', dpi=300, bbox_inches='tight')
        print("Grafik 'boost_inverter_advanced.png' olarak kaydedildi.")
        plt.show()
        
    def print_summary(self):
        """Ozet rapor"""
        print("\n" + "=" * 60)
        print("SIMULASYON SONUC OZETI")
        print("=" * 60)
        print(f"Ortalama Boost Gerilimi    : {self.stats['avg_v_boost']:.2f} V")
        print(f"Cikis RMS Gerilimi         : {self.stats['v_rms_output']:.2f} V")
        print(f"Ortalama Giris Gucu        : {self.stats['avg_power_in']:.2f} W")
        print(f"Ortalama Cikis Gucu        : {self.stats['avg_power_out']:.2f} W")
        print(f"Sistem Verimi              : {self.stats['efficiency']:.2f} %")
        print(f"Total Harmonic Distortion  : {self.stats['thd']:.2f} %")
        print("=" * 60)


# ANA PROGRAM
if __name__ == "__main__":
    # Özel konfigürasyon oluştur
    config = SimulationConfig(
        V_in=12.0,
        V_boost_target=340.0,
        V_rms_target=220.0,
        f_grid=50.0,
        f_pwm=20000.0,
        t_sim=0.1,
        R_load=50.0
    )
    
    # Simülasyon oluştur
    sim = BoostInverterSimulation(config)
    
    # Çalıştır
    sim.run_simulation()
    
    # Sonuçları kaydet
    sim.save_to_csv()
    sim.save_config()
    
    # Görselleştir
    sim.plot_results()
    
    # Özet rapor
    sim.print_summary()
    # Yaman Alparslan