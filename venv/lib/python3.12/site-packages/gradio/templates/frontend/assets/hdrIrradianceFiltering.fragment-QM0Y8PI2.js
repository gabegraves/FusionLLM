import{j as e}from"./index-WSsqGO1o.js";import"./helperFunctions-B1t2y-6j.js";import"./hdrFilteringFunctions-DxPZq_gu.js";import"./index-DDjElQnM.js";import"./svelte/svelte.js";const r="hdrIrradianceFilteringPixelShader",i=`#include<helperFunctions>
#include<importanceSampling>
#include<pbrBRDFFunctions>
#include<hdrFilteringFunctions>
uniform samplerCube inputTexture;
#ifdef IBL_CDF_FILTERING
uniform sampler2D icdfTexture;
#endif
uniform vec2 vFilteringInfo;uniform float hdrScale;varying vec3 direction;void main() {vec3 color=irradiance(inputTexture,direction,vFilteringInfo
#ifdef IBL_CDF_FILTERING
,icdfTexture
#endif
);gl_FragColor=vec4(color*hdrScale,1.0);}`;e.ShadersStore[r]||(e.ShadersStore[r]=i);const a={name:r,shader:i};export{a as hdrIrradianceFilteringPixelShader};
//# sourceMappingURL=hdrIrradianceFiltering.fragment-QM0Y8PI2.js.map
