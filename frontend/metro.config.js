const { getDefaultConfig } = require("expo/metro-config");
const { withNativeWind } = require("nativewind/metro");

const config = getDefaultConfig(__dirname);

config.resolver.extraNodeModules = {
    ...config.resolver.extraNodeModules,
    'react-native-linear-gradient': require.resolve('expo-linear-gradient'),
};

module.exports = withNativeWind(config, { input: './global.css' });
