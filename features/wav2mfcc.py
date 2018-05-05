from praatinterface import PraatLoader

pl = PraatLoader(praatpath = '/usr/bin/praat')

in_dir = '/home/pi/Documents/Project_Jammin/speech_emotion_classifier/wav/wav/carlos/casa_may_24_2016'
#in_dir = '/home/pi/Documents/Project_Jammin/speech_emotion_classifier/test'
out_dir = '/home/pi/Documents/Project_Jammin/speech_emotion_classifier/vectors/carlos/casa_may_24_2016/mfcc'
#out_dir = '/home/pi/Documents/Project_Jammin/speech_emotion_classifier/vectors/live'
text = pl.run_script('/home/pi/Documents/Project_Jammin/speech_emotion_classifier/wav2mfcc.praat', in_dir, out_dir )
#text = pl.run_script('/home/pi/Documents/Project_Jammin/speech_emotion_classifier/formants.praat','/home/pi/Documents/Project_Jammin/speech_recognition/wav/test.wav', 5500, 5)

print text
mats = pl.read_praat_out(text)
print mats
print "Done"
