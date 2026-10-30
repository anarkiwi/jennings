static const unsigned int top_octave[12] = {17820, 18880, 20003, 21192, 22452, 23787,
                                            25202, 26700, 28288, 29970, 31752, 33640};

unsigned int note_freq(unsigned char note) {
  unsigned char base = 72;
  while (note < base) base -= 12;
  unsigned int f = top_octave[note - base];
  for (; base < 72; base += 12) f >>= 1;
  return f;
}
