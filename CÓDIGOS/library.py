import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import signal


def features(signal, timestep):
    coeff, frequency = fft(signal, timestep)
    spectrum = psd(coeff)
    mean = mav(signal)
    variance = var(signal)
    root_mean_square = rms(signal)
    total_power = ttp(spectrum)
    peak_frequency = pf(spectrum, frequency)
    r0 = rsm0(spectrum)
    r1 = rsm1(spectrum, frequency)
    r2 = rsm2(spectrum, frequency)
    r3 = rsm3(spectrum, frequency)
    r4 = rsm4(spectrum, frequency)
    myop_threshold = 0.704
    wamp_threshold = 0.415
    zc_threshold = 0.670
    ssc_threshold = 0.077
    ar_order = 2
    m_apen, r_apen = 1, 0.5
    m_sampen, r_sampen = 1, 0.5
    cc_order = 2
    n_hist = 8
    kmax_higuchi = 6
    l_dfa = 4
    v = 3
    k_mavs = 2
    k_mhw = 3
    n_psr = 20
    llc, ulc, lhc, uhc = 15, 45, 95, 150
    return np.array([apen(signal, m_apen, r_apen), ar(signal, ar_order), aac(signal),
                    cc(signal, cc_order), dfa(signal, mean, l_dfa), damv(signal), dasdv(signal),
                    dar(signal, ar_order), dcc(signal, cc_order), dld(signal), dvarv(signal),
                    dvord(signal, v), fr(spectrum, frequency, llc, ulc, lhc, uhc), hg(signal, kmax_higuchi),
                    hist(signal, n_hist), iemg(signal), IF(r0, r2, r4), katz(signal, timestep),
                    kurt(signal, mean, variance), ld(signal), max(signal), mfl(signal), mean,
                    mavs(signal, k_mavs), mnf(spectrum, frequency), mp(signal, root_mean_square),
                    mnp(spectrum), mdf(spectrum, frequency, total_power), mav1(signal), mav2(signal),
                    mhw(signal, k_mhw), myop(signal, myop_threshold), npk(signal, root_mean_square),
                    peak_frequency, psr(spectrum, total_power, n_psr), ohm(spectrum, frequency),
                    root_mean_square, sampen(signal, m_sampen, r_sampen), m2(signal), ssi(signal),
                    skew(signal, mean, variance), ssc(signal, ssc_threshold), sparse(r0, r2, r4),
                    r0, r1, r2, r3, r4, sd(signal, mean), tm3(signal), tm4(signal),
                    tm5(signal), total_power, variance, vcf(total_power, r1, r2), vord(signal, v),
                    wl(signal), wlr(signal), wamp(signal, wamp_threshold),
                    zc(signal, zc_threshold)])


"""código FFT"""


def fft(signal, timestep):
    return np.fft.rfft(signal), np.fft.rfftfreq(signal.size, timestep)


"""TIME DOMAIN FEATURES"""

"""(1) Integrated Electromyogram"""


def iemg(signal):
    return np.sum(np.absolute(signal))


"""(2) Mean Absolute Value"""


def mav(signal):
    return np.mean(np.absolute(signal))


"""(3) Root Mean Square"""


def rms(signal):
    return np.sqrt(np.dot(signal, signal) / signal.size)


"""(4) Variance"""


def var(signal):
    return np.var(signal)


"""(5) Simple Square Integral"""


def ssi(signal):
    return np.dot(signal, signal)


"""(6) Myopulse Percentage Rate"""


def myop(signal, threshold):
    return np.sum(f(signal, threshold)) / signal.size


"""(7) Waveform length"""


def wl(signal):
    return np.sum(np.absolute(diff(signal)))


"""(8) Difference Absolute Mean Value"""


def damv(signal):
    return np.sum(np.absolute(diff(signal))) / (signal.size - 1)


"""(9) Difference Second order moment"""


def m2(signal):
    return np.dot(diff(signal), diff(signal))


"""(10) Difference Variance Value"""


def dvarv(signal):
    return np.dot(diff(signal), diff(signal)) / (signal.size - 2)


"""(11) Difference Absolute Standard Deviation Value"""


def dasdv(signal):
    return np.sqrt(np.dot(diff(signal), diff(signal)) / (signal.size - 1))


"""(12) Willison Amplitude"""


def wamp(signal, threshold):
    return np.sum(f(np.absolute(diff(signal)), threshold))


"""(13) Log Detector"""


def ld(signal):
    return np.exp(np.sum(np.log(np.absolute(signal))) / signal.size)


"""(14) Modified Mean Absolute Value type 1"""


def mav1(signal):
    return np.dot(np.absolute(signal),
                  np.array([1 if np.floor((signal.size / 4) - 1) < i <= np.ceil(0.75 * signal.size - 1)
                            else 0.5 for i in range(signal.size)]))


"""(15) Modified Mean Absolute Value type 2"""


def mav2(signal):
    return np.dot(signal, np.array([1 if np.floor(0.25 * signal.size - 1) < i <= np.ceil(0.75 * signal.size - 1)
                                    else 4 * (i + 1 - signal.size) / signal.size if i >= signal.size / 4 - 1
                                    else 4 * i / signal.size for i in range(signal.size)]))


"""(16) Absolute Value of the 3th moment"""


def tm3(signal):
    return np.absolute(np.sum(signal * signal * signal) / signal.size)


"""(17) Absolute Value of the 4th moment"""


def tm4(signal):
    return np.sum(signal * signal * signal * signal) / signal.size


"""(18) Absolute Value of the 5th moment"""


def tm5(signal):
    return np.absolute(np.sum(signal * signal * signal * signal * signal) / signal.size)


"""(19) Zero Crossing"""


def zc(signal, threshold):
    return np.sum(np.array([1 if ((signal[i] > 0 > signal[i + 1]) or (signal[i] < 0 < signal[i + 1])) and
                                 np.absolute(signal[i + 1] - signal[i]) >= threshold else 0 for i in
                            range(signal.size - 1)]))


"""(20) Slope Signing Change"""


def ssc(signal, threshold):
    return (np.sum(
        np.array([1 if (((signal[i - 1] < signal[i] > signal[i + 1]) or (signal[i - 1] > signal[i] < signal[i + 1])) and
                        ((np.absolute(signal[i] - signal[i - 1] > threshold)) or
                         (np.absolute(signal[i + 1] - signal[i]) > threshold)))
                  else 0 for i in range(1, signal.size - 1)])))


"""(21) Autoregressive Coefficients"""


def ar(signal, p):
    x = []
    y = []
    for i in range(p, signal.size):
        x_ = []
        for j in range(1, p+1):
            x_.append(signal[i-j])
        y.append(signal[i])
        x.append(x_)
    x = np.array(x)
    y = np.array(y)
    model = LinearRegression()
    model.fit(x,y)
    coefficients = []
    for i in range(model.coef_.size):
        coefficients.append(model.coef_[i])
    coefficients.append(model.intercept_)
    return np.mean(np.array(coefficients))


"""(22) Average Amplitude Change """


def aac(signal):
    return np.sum(np.absolute(diff(signal))) / signal.size


"""(23) Approximate Entropy """


def apen(signal, m, r):
    pos = np.zeros(signal.size - m)
    mat = np.zeros(signal.size - m)
    for i in range(signal.size - m):
        possibles = 0
        matches = 0
        for j in range(signal.size - m):
            k = 0
            stop = True
            while k < m + 1 and stop:
                if k < m - 1 and np.absolute(signal[i + k] - signal[j + k]) > r:
                    stop = False
                elif k == m - 1:
                    if np.absolute(signal[i + k] - signal[j + k]) > r:
                        stop = False
                    else:
                        possibles += 1
                else:
                    if np.absolute(signal[i + k] - signal[j + k]) > r:
                        stop = False
                    else:
                        matches += 1
                k += 1
        pos[i] = possibles
        mat[i] = matches
    return (-1) * np.sum(np.log(mat / pos)) / (signal.size - m)


"""(24) Sample Entropy """


def sampen(signal, m, r):
    pos = np.zeros(signal.size - m)
    mat = np.zeros(signal.size - m)
    for i in range(signal.size - m):
        possibles = 0
        matches = 0
        for j in range(signal.size - m):
            if j != i:
                k, stop = 0, True
                while k < m + 1 and stop:
                    if k < m - 1 and np.absolute(signal[i + k] - signal[j + k]) > r:
                        stop = False
                    elif k == m - 1:
                        if np.absolute(signal[i + k] - signal[j + k]) > r:
                            stop = False
                        else:
                            possibles += 1
                    else:
                        if np.absolute(signal[i + k] - signal[j + k]) > r:
                            stop = False
                        else:
                            matches += 1
                    k += 1
        pos[i] = possibles
        mat[i] = matches
    return (-1) * np.log((np.sum(mat) + 1) / (np.sum(pos)+1))


"""(25) Cepstral Coefficients """


def cc(signal, p):
    x = []
    y = []
    for i in range(p, signal.size):
        x_ = []
        for j in range(1, p + 1):
            x_.append(signal[i - j])
        y.append(signal[i])
        x.append(x_)
    x = np.array(x)
    y = np.array(y)
    model = LinearRegression()
    model.fit(x, y)
    a = []
    for i in range(model.coef_.size):
        a.append(model.coef_[i])
    a.append(model.intercept_)
    c = []
    for i in range(p):
        if len(c) == 0:
            c.append((-1)*a[0])
        else:
            s = 0
            for j in range(p):
                s += (1-j/i)*c[i-(j+1)]*a[p-1]
            c.append((-1)*a[i] + s)
    return np.mean(np.array(c))


"""(26) Kurtosis """


def kurt(signal, mean, variance):
    return (np.sum(
        ((1 / (variance ** 4)) * (signal - mean * np.ones(signal.size)) * (signal - mean * np.ones(signal.size)) *
         (signal - mean * np.ones(signal.size)) * (signal - mean * np.ones(signal.size)))) / signal.size) - 3


"""(27) Skewness """


def skew(signal, mean, variance):
    return np.sum(
        ((1 / (variance ** 3)) * (signal - mean * np.ones(signal.size)) * (signal - mean * np.ones(signal.size)) *
         (signal - mean * np.ones(signal.size)))) / signal.size


"""(28) Katz's Fractal Dimension """


def katz(signal, timestep):
    t = np.array([i*timestep for i in range(signal.size)])
    d = 0
    l = 0
    for i in range(signal.size):
        if i < signal.size - 1:
            l = l +  distance(t[i], signal[i], t[i + 1], signal[i + 1])
        if distance(t[0], signal[0], t[i], signal[i]) > d and i != 0:
            d = distance(t[0], signal[0], t[i], signal[i])
    return np.log(l/signal.size) / (np.log(l/signal.size) + np.log(d / l))


"""(29) Maximum Fractal Length """


def mfl(signal):
    return np.log10(np.sqrt(np.dot(diff(signal), diff(signal))))


"""(30) Histogram """


def hist(signal, n):
    mn, mx = np.min(signal), np.max(signal)
    length = np.ceil(np.absolute(mx - mn)/n)
    intervals = np.array([mn + i*length for i in range(1, n+1)])
    values = np.zeros(n)
    for i in range(signal.size):
        for j in range(intervals.size):
            if signal[i] > intervals[j]:
                values[j-1] += 1
    x = 0
    for i in range(intervals.size-1):
        x += (intervals[i+1] - intervals[i])*values[i]/2
    return x


"""(31) Higuchi's method """


def hg(signal, kmax):
    l = []
    for k in range(1, kmax+1):
        v_m = []
        for m in range(1, k+1):
            s = 0
            for i in range(int(np.floor((signal.size - m)/k))):
                s += np.absolute(signal[m+i*k] - signal[m + (i-1)*k])
            v_m.append((signal.size-1)*s/(np.floor((signal.size-m)/k)*(k**2)))
        l.append(np.mean(np.array(v_m)))
    return np.mean(np.array(l))


"""(32) Detrended Fluctuation Analysis"""


def dfa(signal, mean, L):
    y = signal - mean*np.ones(signal.size)
    x = np.arange(signal.size)
    n = int(np.floor(signal.size/L))
    fit = []
    for i in range(L):
        if i == 0:
            fit.append(np.polyfit(x[:n], y[:n], 2))
        else:
            fit.append(np.polyfit(x[i*n:(i+1)*n], y[i*n:(i+1)*n], 2))
    fit = np.array(fit)
    f = []
    for i in range(L):
        s = 0
        if i == 0:
            for j in range(n):
                s += (y[j] - (fit[i,0]*(x[j]**2) + fit[i,1]*x[j] + fit[i, 2]))**2
        else:
            for j in range(n):
                s += (y[j] - (fit[i, 0] * (x[j] ** 2) + fit[i, 1] * x[j] + fit[i, 2])) ** 2
        f.append(np.sqrt(s/n))
    f = np.array(f)
    return np.mean(np.polyfit(np.log(np.array([i*n for i in range(1,L+1)])), np.log(f), 1)[0])


"""(33) Maximum Amplitude """


def max(signal):
    return np.max(np.absolute(signal))


"""(34) Standard Deviation """


def sd(signal, mean):
    return np.sqrt(np.dot(signal - mean*np.ones(signal.size), signal - mean*np.ones(signal.size))/(signal.size-1))

"""(35) v-Order """


def vord(signal, v):
    return np.power(np.absolute(np.mean(np.power(signal, v))), 1/v)


"""(36) Difference Log Detector """


def dld(signal):
    return np.exp(np.sum(np.log(np.absolute(diff(signal)))) / signal.size)


"""(37) Difference v-Order """


def dvord(signal, v):
    return np.power(np.absolute(np.mean(np.power(diff(signal), v))) , 1/v)


"""(38) Difference Auto-Regressive Coefficients """


def dar(signal, p):
    return ar(diff(signal), p)


"""(39) Difference Cepstral Coefficients """


def dcc(signal, p):
    return cc(diff(signal), p)


"""(40) Waveform Length Ratio """


def wlr(signal):
    return np.log(np.sum(np.absolute(diff(signal)))/np.sum(np.absolute(diff(diff(signal)))))


"""(41) Number of Peaks """


def npk(signal, rms):
    return np.sum(np.array([1 if signal[i] > rms else 0 for i in range(signal.size)]))


"""(42) Mean of Peaks """


def mp(signal, rms):
    vetor = np.array([1 if np.absolute(signal[i]) > rms else 0 for i in range(signal.size)])
    return np.dot(np.absolute(signal),vetor)


"""(43) Mean Absolute Value Slope """


def mavs(signal, k):
    segments = []
    for i in range(k):
        if i == 0:
            segments.append(signal[:int(np.ceil(signal.size/k))])
        elif i == k - 1:
            segments.append(signal[int(np.ceil(signal.size/k)*i):])
        else:
            segments.append(signal[int(np.ceil(signal.size / k)) * i:int(np.ceil(signal.size / k)) * (i+1)])
    segments =np.array(segments)
    return np.mean(np.array([np.mean(segments[i+1])-np.mean(segments[i]) for i in range(k-1)]))


"""(44) Multiple Hamming Windows """


def mhw(signal, k):
    a0 = 0.53836
    w = []
    n = int(signal.size/k)
    for i in range(n):
        w.append(a0 - (1-a0)*np.cos(2*np.pi*i/n))
    w = np.array(w)
    windows = []
    for j in range(k):
        if j == 0:
            windows.append(signal[:n])
        else:
            windows.append(signal[j*n:(j+1)*(n)])
    windows = np.array(windows)
    return np.mean(np.array([np.dot(windows[i]*w, windows[i]*w) for i in range(k)]))


"""FREQUENCY DOMAIN FEATURES"""
"""OBS: Spectrum equivale a P[k]xP*[k] """
"""(1) Mean Frequency"""


def mnf(spectrum, freq):
    return np.dot(spectrum, freq) / np.sum(spectrum)


"""(2) Median Frequency (Detalhes importantes precisam ser verificados aqui)"""


def mdf(spectrum, freq, total_power):
    sum, i = 0, 0
    while i < spectrum.size and sum < total_power / 2:
        sum += spectrum[i]
        i += 1
    if sum - total_power / 2 > 0:
        return freq[i - 1] + 2 * (sum - total_power / 2) / total_power
    return freq[i - 1]


"""(3) Peak Frequency"""


def pf(spectrum, freq):
    return freq[np.argmax(spectrum)]


"""(4) Mean Power"""


def mnp(spectrum):
    return np.mean(spectrum)


"""(5) Total Power"""


def ttp(spectrum):
    return np.sum(spectrum)


"""(6) Root Squared First Order Moment"""


def rsm1(spectrum, freq):
    return np.dot(spectrum, freq)


"""(7) Root Squared Second Order Moment"""


def rsm2(spectrum, freq):
    return np.sqrt(np.dot(spectrum, freq * freq))


"""(8) Root Squared Third Order Moment"""


def rsm3(spectrum, freq):
    return np.power(np.dot(spectrum, freq * freq * freq), 1/3)


"""(9) Root Squared Fourth Order Moment"""


def rsm4(spectrum, freq):
    return np.power(np.dot(spectrum, freq * freq * freq * freq),1/4)


"""(10) Frequency Ratio"""


def fr(spectrum, freq, llc, ulc, lhc, uhc):
    fllc, fulc, flhc, fuhc = True, True, True, True
    for i in range(len(spectrum)):
        if fllc and freq[i] >= llc:
            fllc = False
            illc = i
        if fulc and freq[i] > ulc:
            fulc = False
            iulc = i
        if flhc and freq[i] >= lhc:
            flhc = False
            ilhc = i
        if fuhc and freq[i] > uhc:
            fuhc = False
            iuhc = i
    return np.sum(spectrum[illc:iulc]) / np.sum(spectrum[ilhc:iuhc])


"""(11) Power Spectral Ratio"""


def psr(spectrum, total_power, n):
    return np.sum(spectrum[np.argmax(spectrum) - n:np.argmax(spectrum) + n + 1]) / total_power


"""(12) Variance of Central Frequency"""


def vcf(total_power, r1, r2):
    return (r2 / total_power) - (r1 / total_power) ** 2


"""(13) Power Spectrum Deformation"""


def ohm(spectrum, freq):
    return np.sqrt(rsm2(spectrum, freq)/ttp(spectrum))/(rsm1(spectrum, freq)/ttp(spectrum))


"""(14) Root Squared Zeroth Order Moment """


def rsm0(spectrum):
    return np.sqrt(np.dot(spectrum, np.conjugate(spectrum)))


"""(15) Sparseness """


def sparse(r0, r2, r4):
    return np.log(r0 / np.sqrt(np.absolute((r0 - r2) * (r0 - r4))))


"""(16) Irregularity Factor """


def IF(r0, r2, r4):
    return np.log(r2/np.sqrt(r0*r4))


"""Funções Auxiliares"""


def f(a, threshold):
    return np.array([1 if np.absolute(a[i]) >= threshold else 0 for i in range(a.size)])


def diff(a):
    return np.array([a[i + 1] - a[i] for i in range(a.size - 1)])


def distance(x1, y1, x2, y2):
    return np.sqrt(((x2 - x1) ** 2) + ((y2 - y1) ** 2))


def psd(fourier_transform):
    return np.absolute(fourier_transform)


def butterwoth_lowpass(sg, N, highcut, f=1000):
    sos = signal.butter(N, highcut, 'low', output='sos', fs=f)
    return signal.sosfiltfilt(sos, sg)


def butterwoth_highpass(sg, N, lowcut, f=1000):
    sos = signal.butter(N, lowcut, 'high', output='sos', fs=f)
    return signal.sosfiltfilt(sos, sg)


def butterwoth_bandpass(sg, N, lowcut, highcut, f=1000):
    sos = signal.butter(N, [lowcut, highcut], 'band', output='sos', fs=f)
    return signal.sosfilt(sos, sg)