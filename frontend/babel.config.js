// module.exports = function (api) {
//   api.cache(true);
//   return {
//     presets: ["babel-preset-expo"],
//     presets: [
//       ["babel-preset-expo", { jsxImportSource: "nativewind" }],
//       "nativewind/babel",
//     ],
//     plugins: [["inline-import", { "extensions": [".sql"] }]] 

//   };
// };

// module.exports = function (api) {
//   api.cache(true);
//   return {
//     presets: [
//       ["babel-preset-expo", { 
//         jsxImportSource: "nativewind" 
//       }]
//     ],
//     plugins: [
//       "nativewind/babel",
//       ["inline-import", { 
//         extensions: [".sql"] 
//       }]
//     ]
//   };
// };


module.exports = function (api) {
  api.cache(true);
  return {
    presets: [
      // 1) Expo’s default preset, with NativeWind’s JSX import support
      ["babel-preset-expo", { jsxImportSource: "nativewind" }],
      // 2) NativeWind’s own Babel preset (not a plugin)
      "nativewind/babel"
    ],
    plugins: [
      // Your other plugins (e.g., inline-import, reanimated)
      ["inline-import", { extensions: [".sql"] }],
    ],
  };
};
