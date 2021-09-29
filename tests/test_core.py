import numpy as np
import torch
# TODO: Get rid of absl
from absl.testing import parameterized
from scipy import signal

from rt_ddsp import core


def create_wave_np(batch_size, frequencies, amplitudes, seconds, n_samples):
    """Helper function that synthesizes ground truth harmonic waves with numpy.

    Args:
      batch_size: Number of waves in the batch.
      frequencies: Array of harmonic frequencies in each wave. Shape (n_batch,
        n_time, n_harmonics). Units in Hertz.
      amplitudes: Array of amplitudes for each harmonic. Shape (n_batch, n_time,
        n_harmonics). Units in range 0 to 1.
      seconds: Length of the waves, in seconds.
      n_samples: Length of the waves, in samples.

    Returns:
      wave_np: An array of the synthesized waves. Shape (n_batch, n_samples).
    """
    wave_np = np.zeros([batch_size, n_samples])
    time = np.linspace(0, seconds, n_samples)
    n_harmonics = int(frequencies.shape[-1])
    for i in range(batch_size):
        for j in range(n_harmonics):
            rads_per_cycle = 2.0 * np.pi
            rads_per_sec = rads_per_cycle * frequencies[i, :, j]
            phase = time * rads_per_sec
            wave_np[i, :] += amplitudes[i, :, j] * np.sin(phase)
    return wave_np


class HarmonicSynthTest(parameterized.TestCase):

    def setUp(self):
        """Creates some common default values for the tests."""
        super().setUp()
        self.batch_size = 2
        self.sample_rate = 16000
        self.seconds = 1.0
        self.n_samples = int(self.seconds) * self.sample_rate

    @parameterized.named_parameters(
        ('low_frequency', 2, 62.4, 5, 16000, 2),
        ('large_batch_size', 16, 100, 1, 8000, 0.5),
        ('high_frequency', 1, 2000, 2, 4000, 1.3),
    )
    def test_oscillator_bank_is_accurate(self, batch_size, fundamental_frequency,
                                         n_harmonics, sample_rate, seconds):
        """Test waveforms generated from oscillator_bank.

        Generates harmonic waveforms with tensorflow and numpy and tests that they
        are the same. Test over a range of inputs provided by the parameterized
        inputs.

        Args:
          batch_size: Size of the batch to synthesize.
          fundamental_frequency: Base frequency of the oscillator in Hertz.
          n_harmonics: Number of harmonics to synthesize.
          sample_rate: Sample rate of synthesis in samples per a second.
          seconds: Length of the generated test sample in seconds.
        """
        n_samples = int(sample_rate * seconds)
        seconds = float(n_samples) / sample_rate
        frequencies = fundamental_frequency * np.arange(1, n_harmonics + 1)
        amplitudes = 1.0 / n_harmonics * np.ones_like(frequencies)

        # Create tensors of frequencies and amplitudes for tf function.
        ones = np.ones([batch_size, n_samples, n_harmonics])
        frequency_envelopes = ones * frequencies[np.newaxis, np.newaxis, :]
        amplitude_envelopes = ones * amplitudes[np.newaxis, np.newaxis, :]

        # Create np test signal.
        wav_np = create_wave_np(batch_size, frequency_envelopes,
                                amplitude_envelopes, seconds, n_samples)

        wav_tf = core.oscillator_bank(
            frequency_envelopes, amplitude_envelopes, sample_rate=sample_rate)
        pad = 10  # Ignore edge effects.
        np.testing.assert_array_almost_equal(wav_np[pad:-pad], wav_tf[pad:-pad].numpy())

    @parameterized.named_parameters(
        ('sum_sinusoids', True),
        ('no_sum_sinusoids', False),
    )
    def test_oscillator_bank_shape_is_correct(self, sum_sinusoids):
        """Tests that sum_sinusoids reduces the last dimension."""
        frequencies = np.array([1.0, 1.5, 2.0]) * 400.0
        amplitudes = np.ones_like(frequencies)

        # Create tensors of frequencies and amplitudes for tf function.
        ones = np.ones([self.batch_size, self.n_samples, 3])
        frequency_envelopes = ones * frequencies[np.newaxis, np.newaxis, :]
        amplitude_envelopes = ones * amplitudes[np.newaxis, np.newaxis, :]

        wav_tf = core.oscillator_bank(frequency_envelopes,
                                      amplitude_envelopes,
                                      sample_rate=self.sample_rate,
                                      sum_sinusoids=sum_sinusoids)
        if sum_sinusoids:
            expected_shape = [self.batch_size, self.n_samples]
        else:
            expected_shape = [self.batch_size, self.n_samples, 3]
        self.assertListEqual(expected_shape, list(wav_tf.shape))

    @parameterized.named_parameters(
        ('low_sample_rate', 4000),
        ('16khz', 16000),
        ('cd_quality', 44100),
    )
    def test_silent_above_nyquist(self, sample_rate):
        """Tests that no freqencies above nyquist (sample_rate/2) are created."""
        nyquist = sample_rate / 2
        frequencies = np.array([1.1, 1.5, 2.0]) * nyquist
        amplitudes = np.ones_like(frequencies)

        # Create tensors of frequencies and amplitudes for tf function.
        ones = np.ones([self.batch_size, self.n_samples, 3])
        frequency_envelopes = ones * frequencies[np.newaxis, np.newaxis, :]
        amplitude_envelopes = ones * amplitudes[np.newaxis, np.newaxis, :]

        wav_tf = core.oscillator_bank(
            frequency_envelopes, amplitude_envelopes, sample_rate=sample_rate)
        wav_np = np.zeros_like(wav_tf)
        np.testing.assert_array_almost_equal(wav_np, wav_tf)

    @parameterized.named_parameters(
        ('low_frequency', 2, 20, 0.1, 100),
        ('many_frames', 1, 100, 0.2, 1000),
        ('high_frequency', 4, 2000, 0.5, 100),
    )
    def test_harmonic_synthesis_is_accurate_one_frequency(self, batch_size,
                                                          fundamental_frequency,
                                                          amplitude, n_frames):
        """Tests generating a single sine wave with different frame parameters.

        Generates sine waveforms with tensorflow and numpy and tests that they are
        the same. Test over a range of inputs provided by the parameterized
        inputs.

        Args:
          batch_size: Size of the batch to synthesize.
          fundamental_frequency: Base frequency of the oscillator in Hertz.
          amplitude: Amplitude of each harmonic in the waveform.
          n_frames: Number of amplitude envelope frames.
        """
        frequencies = fundamental_frequency * np.ones([batch_size, n_frames, 1])
        amplitudes = amplitude * np.ones([batch_size, n_frames, 1])

        frequencies_np = fundamental_frequency * np.ones(
            [batch_size, self.n_samples, 1])
        amplitudes_np = amplitude * np.ones([batch_size, self.n_samples, 1])

        # Create np test signal.
        wav_np = create_wave_np(batch_size, frequencies_np, amplitudes_np,
                                self.seconds, self.n_samples)

        wav_tf = core.harmonic_synthesis(
            frequencies,
            amplitudes,
            n_samples=self.n_samples,
            sample_rate=self.sample_rate)
        pad = self.n_samples // n_frames  # Ignore edge effects.
        np.testing.assert_array_almost_equal(wav_np[pad:-pad], wav_tf[pad:-pad])

    @parameterized.named_parameters(
        ('one_harmonic', 1),
        ('twenty_harmonics', 20),
        ('forty_harmonics', 40),
    )
    def test_harmonic_synthesis_is_accurate_multiple_harmonics(self, n_harmonics):
        """Tests generating a harmonic waveform with varying number of harmonics.

        Generates waveforms with tensorflow and numpy and tests that they are
        the same. Test over a range of inputs provided by the parameterized
        inputs.

        Args:
          n_harmonics: Number of harmonics to synthesize.
        """
        fundamental_frequency = 440.0
        amp = 0.1
        n_frames = 100

        harmonic_shifts = np.abs(np.random.randn(1, 1, n_harmonics))
        harmonic_distribution = np.abs(np.random.randn(1, 1, n_harmonics))

        frequencies_tf = fundamental_frequency * np.ones(
            [self.batch_size, n_frames, 1])
        amplitudes_tf = amp * np.ones([self.batch_size, n_frames, 1])
        harmonic_shifts_tf = np.tile(harmonic_shifts, [1, n_frames, 1])
        harmonic_distribution_tf = np.tile(harmonic_distribution, [1, n_frames, 1])

        # Create np test signal.
        frequencies_np = fundamental_frequency * np.ones(
            [self.batch_size, self.n_samples, 1])
        amplitudes_np = amp * np.ones([self.batch_size, self.n_samples, 1])
        frequencies_np = frequencies_np * harmonic_shifts
        amplitudes_np = amplitudes_np * harmonic_distribution
        wav_np = create_wave_np(self.batch_size, frequencies_np, amplitudes_np,
                                self.seconds, self.n_samples)

        wav_tf = core.harmonic_synthesis(
            torch.from_numpy(frequencies_tf),
            torch.from_numpy(amplitudes_tf),
            torch.from_numpy(harmonic_shifts_tf),
            torch.from_numpy(harmonic_distribution_tf),
            n_samples=self.n_samples,
            sample_rate=self.sample_rate)
        pad = self.n_samples // n_frames  # Ignore edge effects.
        np.testing.assert_array_almost_equal(wav_np[pad:-pad], wav_tf[pad:-pad])


class FiniteImpulseResponseTest(parameterized.TestCase):

    def setUp(self):
        """Creates some common default values for the tests."""
        super().setUp()
        self.audio_size = 1000
        self.audio = np.random.randn(1, self.audio_size).astype(np.float32)

    @parameterized.named_parameters(
        ('ir_less_than_audio', 1000, 10),
        ('audio_less_than_ir', 10, 100),
    )
    def test_fft_convolve_is_accurate(self, audio_size, impulse_response_size):
        """Tests convolving signals using fast fourier transform (fft).

        Generate random signals and convolve using fft. Compare outputs to the
        implementation in scipy.signal.

        Args:
          audio_size: Size of the audio to convolve.
          impulse_response_size: Size of the impulse response to convolve.
        """

        # Create random signals to convolve.
        audio = np.ones([1, audio_size]).astype(np.float32)
        impulse_response = np.ones([1, impulse_response_size]).astype(np.float32)

        output_tf = core.fft_convolve(
            torch.from_numpy(audio), torch.from_numpy(impulse_response),
            padding='valid', delay_compensation=0)[0]

        output_np = signal.fftconvolve(audio[0], impulse_response[0])

        difference = torch.from_numpy(output_np) - output_tf
        total_difference = np.abs(difference).mean()
        threshold = 1e-3
        self.assertLessEqual(total_difference, threshold)

    @parameterized.named_parameters(
        ('unity_gain', 1.0),
        ('reduced_gain', 0.1),
    )
    def test_delay_compensation_corrects_group_delay(self, gain):
        """Test automatically compensating for group delay of linear phase filter.

        Genearate filters to shift entire signal by a constant gain. Test that
        filtered signal is in phase with original signal.

        Args:
          gain: Amount to scale the input signal.
        """
        # Create random signal to filter.
        output_np = gain * self.audio[0]
        n_frequencies = 1025
        window_size = 257

        magnitudes = gain * torch.ones([1, n_frequencies])
        impulse_response = core.frequency_impulse_response(magnitudes, window_size)
        output_tf = core.fft_convolve(torch.from_numpy(self.audio), impulse_response,
                                      padding='same')[0]

        difference = output_np - output_tf.numpy()
        total_difference = np.abs(difference).mean()
        threshold = 1e-3
        self.assertLessEqual(total_difference, threshold)

    def test_fft_convolve_checks_batch_size(self):
        """Tests fft_convolve() raises error for mismatched batch sizes."""
        # Create random signals to convolve with different batch sizes.
        impulse_response = torch.cat(
            [torch.from_numpy(self.audio), torch.from_numpy(self.audio)], dim=0)

        with self.assertRaises(ValueError):
            _ = core.fft_convolve(torch.from_numpy(self.audio), impulse_response)

    @parameterized.named_parameters(
        ('same', 'same'),
        ('valid', 'valid'),
    )
    def test_fft_convolve_allows_valid_padding_arguments(self, padding):
        """Tests fft_convolve() runs for valid padding names."""
        result = core.fft_convolve(torch.from_numpy(self.audio),
                                   torch.from_numpy(self.audio), padding=padding)
        self.assertEqual(result.shape[0], 1)

    @parameterized.named_parameters(
        ('no_name', ''),
        ('bad_name', 'saaammmeee'),
    )
    def test_fft_convolve_disallows_invalid_padding_arguments(self, padding):
        """Tests fft_convolve() raises error for wrong padding name."""
        with self.assertRaises(ValueError):
            _ = core.fft_convolve(torch.from_numpy(self.audio),
                                  torch.from_numpy(self.audio), padding=padding)

    @parameterized.named_parameters(
        ('more_frames_than_timesteps', 1010),
        ('not_even_multiple', 999),
    )
    def test_fft_convolve_checks_number_of_frames(self, n_frames):
        """Tests fft_convolve() raises error for invalid number of framess."""
        # Create random signals to convolve with same batch sizes.
        impulse_response = torch.randn([1, n_frames, self.audio_size])
        with self.assertRaises(ValueError):
            _ = core.fft_convolve(torch.from_numpy(self.audio), impulse_response)

    @parameterized.named_parameters(
        ('no_window', 2048, 0),
        ('typical_window', 2048, 257),
        ('atypical_window', 1024, 22),
        ('window_bigger', 1024, 2048),
    )
    def test_frequency_impulse_response_gives_correct_size(self, fft_size, window_size):
        """Tests generating impulse responses from a list of magnitudes.

        The output size should be determined by the window size, or fft_size if
        window size < 1.

        Args:
          fft_size: Size of the fft that generated the magnitudes.
          window_size: Size of window to apply to inverse fft.
        """
        # Create random signals to convolve.
        n_frequencies = fft_size // 2 + 1
        magnitudes = torch.rand((1, n_frequencies))

        impulse_response = core.frequency_impulse_response(magnitudes, window_size)

        target_size = fft_size
        if target_size > window_size >= 1:
            target_size = window_size
            is_even = target_size % 2 == 0
            target_size -= int(is_even)

        impulse_response_size = int(impulse_response.shape[-1])
        self.assertEqual(impulse_response_size, target_size)

    @parameterized.named_parameters(
        ('no_frames_no_window', 1025, 0, 0),
        ('no_frames_window', 1025, 0, 257),
        ('single_frame', 513, 1, 257),
        ('non_divisible_frames', 513, 13, 257),
        ('max_frames', 513, 1000, 257),
    )
    def test_frequency_filter_gives_correct_size(self, n_frequencies, n_frames,
                                                 window_size):
        """Tests filtering signals with frequency sampling method.

        Generate random signals and magnitudes and filter using fft_convolve().

        Args:
          n_frequencies: Number of magnitudes.
          n_frames: Number of frames for a time-varying filter.
          window_size: Size of window for generating impulse responses.
        """
        # Create transfer function.
        if n_frames > 0:
            magnitudes = np.random.uniform(size=(1, n_frames,
                                                 n_frequencies)).astype(np.float32)
        else:
            magnitudes = np.random.uniform(size=(1, n_frequencies)).astype(np.float32)

        print(self.audio.shape, magnitudes.shape)
        audio_out = core.frequency_filter(
            torch.from_numpy(self.audio),
            torch.from_numpy(magnitudes), window_size=window_size, padding='same')

        audio_out_size = int(audio_out.shape[-1])
        self.assertEqual(audio_out_size, self.audio_size)
