import React, { useState } from 'react';
import { View, Text, Image, ScrollView, TouchableOpacity, Animated, Dimensions } from 'react-native';
import { Ionicons } from '@expo/vector-icons';
import { images } from '../constants';
import { prayerStyles } from '../styles/prayerStyles';

export default function DhuhrWomen() {
  const [currentStep, setCurrentStep] = useState(0);
  const fadeAnim = new Animated.Value(1);

  const windowWidth = Dimensions.get('window').width;

  const dhuhrSteps = [
    {
      title: "Sunnah First Two Rakats",
      description: "Begin with 4 Sunnah rakats (prayed in sets of 2)",
      details: [
        "Make Niyyah for first 2 Sunnah rakats",
        "Say Takbir keeping hands below shoulders",
        "Place hands on chest, right over left",
        "Keep feet together and body compact",
        "Recite Thana quietly",
        "Recite Ta'awwuz and Tasmiah",
        "Recite Surah Al-Fatiha",
        "Recite another Surah",
        "Maintain modest posture throughout"
      ],
      image: images.niyyah
    },
    {
      title: "Sunnah Second Two Rakats",
      description: "Complete the remaining 2 Sunnah rakats",
      details: [
        "Stand and make Niyyah for next 2 rakats",
        "Keep movements controlled and gentle",
        "Maintain proper covering throughout",
        "Recite Al-Fatiha and another Surah in both rakats",
        "Complete all positions modestly",
        "End with Tashahhud and Salam"
      ],
      image: images.qiyam
    },
    {
      title: "Farz First Rakat",
      description: "Begin 4 Farz rakats of Dhuhr",
      details: [
        "Make new Niyyah for 4 Farz rakats",
        "Keep hands below shoulders for Takbir",
        "Place hands on chest, right over left",
        "Recite Thana, Ta'awwuz, Tasmiah",
        "Recite Surah Al-Fatiha",
        "Recite another Surah",
        "For Ruku, bend only as needed",
        "Keep arms close to sides in Ruku"
      ],
      image: images.takbeer
    },
    {
      title: "Farz Second Rakat",
      description: "Complete second Farz rakat",
      details: [
        "Stand gracefully for second Rakat",
        "Keep movements contained and modest",
        "Recite Al-Fatiha and another Surah",
        "For Ruku and Sujood, maintain compact position",
        "Sit for Qaida-e-Oola (first sitting)",
        "Recite At-Tahiyyat quietly",
        "Rise carefully for third Rakat"
      ],
      image: images.ruku
    },
    {
      title: "Farz Third Rakat",
      description: "Continue with third Farz rakat",
      details: [
        "Stand with composed posture",
        "Recite only Surah Al-Fatiha",
        "No additional Surah needed",
        "Keep arms close in Ruku",
        "In Sujood, keep limbs close to body",
        "Maintain modest positioning",
        "Rise gently for fourth Rakat"
      ],
      image: images.sajda
    },
    {
      title: "Farz Fourth Rakat",
      description: "Complete the Farz prayer",
      details: [
        "Recite only Surah Al-Fatiha",
        "Keep movements gentle and controlled",
        "In final sitting, keep feet to right side",
        "Recite At-Tahiyyat",
        "Recite Durood Ibrahim",
        "Recite closing duas",
        "Complete with gentle Salam to both sides"
      ],
      image: images.tashahhud
    },
    {
      title: "Final Sunnah",
      description: "Complete 2 final Sunnah rakats",
      details: [
        "Make Niyyah for 2 final Sunnah rakats",
        "Maintain modest positioning throughout",
        "Keep movements gentle and controlled",
        "Recite Al-Fatiha and another Surah in both rakats",
        "Complete with Tashahhud and Salam",
        "Make personal duas while maintaining composure"
      ],
      image: images.tasleem
    }
  ];

  const handleNext = () => {
    if (currentStep < dhuhrSteps.length - 1) {
      Animated.sequence([
        Animated.timing(fadeAnim, {
          toValue: 0,
          duration: 200,
          useNativeDriver: true,
        }),
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true,
        })
      ]).start();
      setCurrentStep(currentStep + 1);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 0) {
      Animated.sequence([
        Animated.timing(fadeAnim, {
          toValue: 0,
          duration: 200,
          useNativeDriver: true,
        }),
        Animated.timing(fadeAnim, {
          toValue: 1,
          duration: 200,
          useNativeDriver: true,
        })
      ]).start();
      setCurrentStep(currentStep - 1);
    }
  };

  return (
    <View style={prayerStyles.container}>
      <View style={prayerStyles.navigation}>
        <TouchableOpacity 
          onPress={handlePrevious}
          disabled={currentStep === 0}
          style={[prayerStyles.navButton, currentStep === 0 && prayerStyles.disabledButton]}
        >
          <Ionicons name="chevron-back" size={24} color="#16a34a" />
        </TouchableOpacity>
        
        <Text style={prayerStyles.stepText}>
          Step {currentStep + 1} of {dhuhrSteps.length}
        </Text>
        
        <TouchableOpacity 
          onPress={handleNext}
          disabled={currentStep === dhuhrSteps.length - 1}
          style={[prayerStyles.navButton, currentStep === dhuhrSteps.length - 1 && prayerStyles.disabledButton]}
        >
          <Ionicons name="chevron-forward" size={24} color="#16a34a" />
        </TouchableOpacity>
      </View>

      <ScrollView style={prayerStyles.scrollView}>
        <Animated.View style={{ opacity: fadeAnim }}>
          <Text style={prayerStyles.title}>
            {dhuhrSteps[currentStep].title}
          </Text>
          
          <Image 
            source={dhuhrSteps[currentStep].image}
            style={[prayerStyles.image, { width: windowWidth - 32 }]}
            resizeMode="cover"
          />
          
          <View style={prayerStyles.descriptionContainer}>
            <Text style={prayerStyles.descriptionText}>
              {dhuhrSteps[currentStep].description}
            </Text>
          </View>

          <View style={prayerStyles.detailsContainer}>
            {dhuhrSteps[currentStep].details.map((detail, index) => (
              <View key={index} style={prayerStyles.detailRow}>
                <View style={prayerStyles.bullet} />
                <Text style={prayerStyles.detailText}>{detail}</Text>
              </View>
            ))}
          </View>
        </Animated.View>
      </ScrollView>

      <View style={prayerStyles.progressContainer}>
        {dhuhrSteps.map((_, index) => (
          <View
            key={index}
            style={[
              prayerStyles.progressDot,
              index === currentStep && prayerStyles.activeDot
            ]}
          />
        ))}
      </View>
    </View>
  );
}