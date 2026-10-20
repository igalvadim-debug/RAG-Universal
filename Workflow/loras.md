# LoRA Übersicht mit Aktivierungswörtern (Trigger-Words)

Stand: 2026-10-10. Quellen: SHA256-Hash-Abgleich mit CivitAI (API), früherer Abgleich in dieser Datei (loras.md), Websuche (HuggingFace/CivitAI/ComfyUI), Dateiname.
Ordner: D: = D:\ComfyUiCuda\models\loras, C: = C:\ai_diffusion\models\loras.
Aktivierungswort „–“ = kein Trigger nötig oder nicht auffindbar (Style-/Enhancer-/Utility-/Speed-LoRAs). Stärke „–“ = nicht bekannt.
Rohdaten (alle Trigger, Hashes, Links): D:\claudeAgent\RAG\workflow\lora_triggers\loras_raw.json

Gesamt: 151 LoRAs, davon 57 mit Aktivierungswort.


## D: ltx2

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| bfs_head_swap_v1_ltx2_ic_lora_6000.safetensors | – | 1.0 | nicht gefunden |
| cowgirl-ride-withfuta.safetensors | cowgirl ride, futa ride | 1.0 | loras.md (Hash-Abgleich) – LTXV 2.3 - I2V Cowgirl ride + Futa ride / v1.0 [LTXV 2.3] |
| DaSiWa_LTX23_NSFW_Bodyphysics_Fluid_Motion_Enhancer_v01.safetensors | D4S1W4_NSFWSME | 0.5 | loras.md (Hash-Abgleich) – NSFW Body Physics Fluid Motion Enhancer / Lora / LTX2.3 / LTX2.3 - v1.0 [LTXV 2.3] |
| epic_cumshots-V1-LTX_2_3-CUMSH0T.safetensors | CUMSH0T | 0.6 | loras.md (Hash-Abgleich) – Epic Cumshots / LTX V1 [LTXV 2.3] |
| Fantasy_Painterly.safetensors | D4rkP41nt3r | 1.0 | loras.md (Hash-Abgleich) – LTX 2.3 - Fantasy Painterly / v1.0 [LTXV 2.3] |
| Fantasy_Realism.safetensors | – | – | CivitAI (kein Trigger) – LTX 2.3 - Fantasy Realism / FantasyRealism_v1 [LTXV 2.3] |
| id-lora-celebvhq-ltx2.3.safetensors | (kein Trigger — ID-LoRA, Referenzbild-gesteuert) | 1.0 | loras.md (Hash-Abgleich) – HF: AviadDahan/LTX-2.3-ID-LoRA-CelebVHQ-3K |
| lightx2v_I2V_14B_480p_cfg_step_distill_rank128_bf16.safetensors | – | 1.0 | CivitAI (kein Trigger) – Lightning Lora, massive speed up for Wan2.1 / Wan2.2 made by Lightx2v / Kijai / Self-Forcing 14B I2V R128 [Wan Video 14B i2v 480p] |
| lightx2v_I2V_14B_480p_cfg_step_distill_rank64_bf16.safetensors | – | 1.0 | CivitAI (kein Trigger) – lightlora wan / lightlora wan [Wan Video 14B i2v 720p] |
| ltx-2-19b-distilled-lora_resized_dynamic_fro09_avg_rank_175_fp8.safetensors | – | – | nicht gefunden |
| ltx-2-19b-ic-lora-detailer.safetensors | – | 0.4 | CivitAI (kein Trigger) – LTX2 - IC LORA DETAILER / LTX2 - IC LORA DETAILER [LTXV] |
| ltx-2-2.3-i2v-nsfw-furry-multi-purpose-sex-lora.safetensors | – | 0.7 | CivitAI (kein Trigger) – LTX 2/2.3 [I2V] NSFW (+furry) - Multi purpose sex lora / Merge v1 x0.5, v2 x0.7 [LTXV 2.3] |
| LTX-2.3-22b-AV-LoRA-talking-head-v1_OHWXPERSON.safetensors | OHWXPERSON (Vermutung, ungeprüft) | 1.0 | loras.md (Hash-Abgleich) |
| ltx-2.3-22b-distilled-1.1_lora-dynamic_fro09_avg_rank_111_bf16.safetensors | – | 0.6 | nicht gefunden |
| ltx-2.3-22b-distilled-lora-1.1_fro90_ceil36.safetensors | – | 0.5 | nicht gefunden |
| ltx-2.3-22b-distilled-lora-384.safetensors | – | 1.0 | CivitAI (kein Trigger) – LTX-2.3 / ltx2.3-22b-distilled-lora [LTXV 2.3] |
| ltx-2.3-22b-ic-lora-ingredients-0.9.safetensors | – | 0.9 | nicht gefunden |
| ltx-2.3-22b-ic-lora-motion-track-control-ref0.5.safetensors | Ultra-cinematic continuous shot with precise motion tracking, camera smoothly follows the main subject | 0.5 | loras.md (Hash-Abgleich) |
| ltx-2.3-22b-ic-lora-union-control-ref0.5.safetensors | – | 0.5 | nicht gefunden |
| ltx-2.3-22b_text_embedding_projection.safetensors | – | 1.0 | nicht gefunden |
| LTX-2.3-danceV2.comfy.safetensors | girl group dance | 1.0 | loras.md (Hash-Abgleich) – LTX-2.3-dance / v2.0 [LTXV 2.3] |
| ltx-2.3-dollyforward_lora.safetensors | Ultra-cinematic dolly-forward shot, camera moves smoothly forward toward the main subject | 1.0 | loras.md (Hash-Abgleich) |
| LTX-2.3-Henshin.safetensors | Henshin transformation | 1.0 | loras.md (Hash-Abgleich) – LTX-2.3-transformation / Henshin transformation [LTXV 2.3] |
| LTX-2.3-ID-LoRA-TalkVid-3K.safetensors | (kein Trigger — ID-LoRA, Referenzbild-gesteuert) | 1.0 | loras.md (Hash-Abgleich) – HF: AviadDahan/LTX-2.3-ID-LoRA-TalkVid-3K |
| LTX-2.3-Latin dance.safetensors | dancing latin dance | 1.15 | loras.md (Hash-Abgleich) – LTX-2.3-Latin dance / v1.0 [LTXV 2.3] |
| LTX-2.3-Licon-MSR-V1.safetensors | (kein Trigger, braucht ComfyUI-Licon-MSR Node) | 1.0 | LiconStudio MSR |
| LTX-2.3-OmniNFT-RL-Lora_bf16.safetensors | – | 1.2 | nicht gefunden |
| ltx-2.3-Sound_besser_foley-400-steps.safetensors | – | 1.0 | nicht gefunden |
| LTX2-i2v-OralSuite.safetensors | – | 0.5 | CivitAI (kein Trigger) – LTX2 - Oral Suite / i2v [LTXV2] |
| ltx2-squish_trigger word_squish it.safetensors | squish it | – | Dateiname + HF Comfy-Org/ltx-2 (Squish-LoRA) – HF: ovi054/LTX-2-19b-Squish-LoRA |
| LTX2.3-IC-LORA-Dual-Character.safetensors | – | 0.8 | CivitAI (kein Trigger) – LTX2.3-IC-LORA-Dual-Character / v0.1 [LTXV 2.3] |
| ltx2.3-ic-subtitles-remove-general.safetensors | – | 1.0 | nicht gefunden |
| Ltx2.3-Licon-VBVR-I2V-96000-R32.safetensors | – | 1.0 | nicht gefunden |
| LTX2.3-MysticXXX.safetensors | – | 1.0 | CivitAI (kein Trigger) – [LTX 2.3] Mystic XXX / v1.0 [LTXV 2.3] |
| ltx2.3-transition_zhuanchang.safetensors | zhuanchang | 1.0 | loras.md (Hash-Abgleich) – valiantcat LTX-2.3 Transition LORA / v1.0 [LTXV 2.3] |
| LTX2.3_Crisp_Enhance.safetensors | – | 1.0 | CivitAI (kein Trigger) – LTX 2.3 - Enhancers / Crisp Enhance [LTXV 2.3] |
| LTX2.3_DMD_hybrid_v2.safetensors | – | 1.0 | CivitAI (kein Trigger) – LTX2.3_DMD_hybrid_v2.safetensors / v1.0 [LTXV 2.3] |
| ltx2.3_fpv_motion.safetensors | fpv | 1.0 | loras.md (Hash-Abgleich) |
| ltx2.3_upscale_ic-lora_06250.safetensors | – | 1.0 | nicht gefunden |
| ltx23_inpaint_rank128_v1.safetensors | – | 1.0 | CivitAI (kein Trigger) – LTX LoRAs / LTX-2.3 INP (T2V) [LTXV 2.3] |
| LTX23_Obscura_Remova_v1.safetensors | Remove the object from the foreground. | 1.25 | CivitAI – Obscura Remova / LTX23 V2V v1 [LTXV 2.3] |
| ltx23_srx_scififilm_lora.safetensors | srx_scififilm | 1.0 | loras.md (Hash-Abgleich) – LTX-2.3 Sci-Fi Cinema / v1.0 [LTXV 2.3] |
| ltx23_zoomout_z00m047.safetensors | z00m047 | 0.85 | loras.md (Hash-Abgleich) – (LTX-2.3 Lora) - Camera Zoom Out (I2V) / v2.3 [LTXV2] |
| ltx25_turbo_distill_r64.safetensors | – | – | CivitAI (kein Trigger) – LTX-2.5 Turbo 8-Step Distill LoRA / v1.0 [LTXV 2.5] |
| LTX_10Eros_LoRA_fro99-avgrank77.safetensors | – | 1.0 | HuggingFace (kein Trigger im README) – HF: maximsobolev275/LTX-10Eros-LoRA-r768 |
| LTX_SulphurEXP_LoRA_fro99-avgrank105.safetensors | – | – | HuggingFace (kein Trigger im README) – HF: maximsobolev275/LTX-SulphurExperimental-LoRA-Optimized |
| ltxdeepthroat_v01.safetensors | LTXdeepthroat | 0.5 | loras.md (Hash-Abgleich) – Daring's Deepthroat H3/LTX2.3 / LTXV 2.3 T2V/I2V v0.1 [LTXV 2.3] |
| m4crom4sti4-huge-natural-breasts-ltx2.3-k3nk.safetensors | m4crom4sti4 | 1.0 | loras.md (Hash-Abgleich) – M4CROM4STI4 - Huge Natural Breasts - LTX2.3 - K3NK / v1.0 [LTXV 2.3] |
| motion_stabilizer_ltx_lora_v2.safetensors | – | – | CivitAI (kein Trigger) – Better Human Motion Minimax H3 / v2.0-LTX23 [LTXV 2.3] Training-Tags: mvmt |
| PENISlora_sulfter_i2v-step00008500.comfy.safetensors | PENISLORA | – | CivitAI – Penis Lora - LTX 2.3 SULPHUR LORA / I2V v1 [LTXV 2.3] |
| sexLoraLTX23Penile_Praxis_V4.safetensors | – | 1.0 | CivitAI (kein Trigger) – Penile Praxis (General NSFW) (LTX-2) (T2V + I2V) / v4.0 (LTX2.3) [LTXV 2.3] |
| sfbehind_LTX2_3_v0_1.safetensors | sfbehind | 0.9 | loras.md (Hash-Abgleich) – Sex from behind (Facing Cam) / LTX-2.3 v1 [LTXV 2.3] |
| Singularity-LTX-2.3_OmniCine_V1.safetensors | 无字幕 | – | CivitAI – Singularity-LTX-2.3_OmniCine_V1 / V1 official version [LTXV 2.3] |
| Western-comic-a-semi_1_LTX23_v1.safetensors | a-semi | – | CivitAI – Western comic semirrealistic 2.5D style LTX 2.3 edition / v1.0 [LTXV 2.3] |

## C: (root)

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| gemma-3-12b-it-abliterated_lora_rank64_bf16.safetensors | – | 1.0 | CivitAI (kein Trigger) – GEMMA-3-12b-it-abliterated_lora_rank64_bf16 / v1.0 [LTXV 2.3] |
| ltx-2-19b-lora-camera-control-static.safetensors | – | – | HuggingFace (kein Trigger im README) – HF: Lightricks/LTX-2-19b-LoRA-Camera-Control-Static |
| nsfw_lora_wan_1.3b.safetensors | (kein Trigger) | – | NSFW-API/NSFW_Wan_1.3b, kein Trigger angegeben |
| pussydiffusion-f2-klein-9b_v2.safetensors | pussy / vagina | – | CivitAI – PussyDiffusion General - Flux2.Klein [9B] / v2.0 [Flux.2 Klein 9B] |
| pussydiffusion-shaved_innie-f2-klein-9b_v1.safetensors | – | – | CivitAI (kein Trigger) – PussyDiffusion Innie - Flux2.Klein [9B] / v1.0 [Flux.2 Klein 9B] |
| Sex_helper_for_NSFW_Wan_1.3b.safetensors | – | – | CivitAI (kein Trigger) – Sex helper for NSFW Wan 1.3b / v0.4 [Wan Video 1.3B t2v] |

## C: Krea2

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| KREA2_D33P_V2.safetensors | D33P | – | CivitAI – [KREA 2] Deepthroat / v2.0 [Krea 2] |
| krea2_darkbrush.safetensors | monochrome ink wash style | – | CivitAI – Krea 2 Official Loras / darkbrush [Krea 2] |
| Krea2_HMNSFW_AIO.safetensors | HMNSFW | – | CivitAI – Krea2 AIO NSFW LoRA / v1.0 [Krea 2] |
| krea2_identity_edit_v1_1.safetensors | – | – | CivitAI (kein Trigger) – Krea 2 Identity Edit / v1.1 [Krea 2] |
| krea2_turbo_2step_rank_64_lora.safetensors | – | – | HuggingFace (kein Trigger im README) – HF: lvladikov/Krea2-Turbo-Distill-2step-LoRA |
| krea2_turbo_lora_rank_64_bf16.safetensors | – | 1.0 | CivitAI (kein Trigger) – Krea 2 Raw Turbo Extracted Lora / krea2_turbo_lora_rank_64 [Krea 2] |
| krea2filterbypass.safetensors | – | – | CivitAI (kein Trigger) – Krea2FilterBypass / 2vector [Krea 2] |
| Krea2NSFWStaerke1.2_1.5V4.safetensors | – | – | CivitAI (kein Trigger) – Krea 2 NSFW V4 / v4.3_EXP [Krea 2] |

## C: ltx2.3

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| LTX-2.3-Licon-MSR-V2.safetensors | (kein Trigger, braucht ComfyUI-Licon-MSR Node) | – | LiconStudio MSR V2 |
| Singularity-LTX-2.3_OmniCine_V1.safetensors | 无字幕 | – | CivitAI – Singularity-LTX-2.3_OmniCine_V1 / V1 official version [LTXV 2.3] |

## C: qwen2.1

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| Qwen-Image-2.1-viggle-turbo-4step-lora-r64.safetensors | – | – | CivitAI (kein Trigger) – [ Accelerator LoRAs ] Applicable to QW21 / Minimax H3 / HYVideo TURBO/LIGHTNING/DMD/FAST / TURBO(Viggle)QW21 v0.2.1 [Qwen 2.1] |
| Qwen-Image-2.1-viggle-turbo-v0.2-5step-lora-r256.safetensors | – | – | HuggingFace (kein Trigger im README) – HF: Viggle/Qwen-Image-2.1-viggle-turbo |

## D: (root)

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| animatediff_lightning_4step_diffusers.safetensors | – | – | HuggingFace (kein Trigger im README) – HF: ByteDance/AnimateDiff-Lightning |
| dmad_minimax_h3_4step_full_critic.safetensors | – | – | nicht gefunden |
| eugen1024lora.safetensors | – | – | nicht gefunden |
| fluxgym_06.01.26_sweta_sweta.safetensors | – | – | nicht gefunden – Training-Tags: and a clear blue sky., trees, buildings |
| ip-adapter-faceid-plus_sd15_lora.safetensors | – | – | HuggingFace (kein Trigger im README) – HF: h94/IP-Adapter-FaceID |
| ip-adapter-faceid-plusv2_sd15_lora.safetensors | – | – | CivitAI (kein Trigger) – ip-adapter-faceid-plusv2 / faceid-plusv2_sd1.5 [SD 1.5] |
| ip-adapter-faceid_sd15_lora.safetensors | – | – | HuggingFace (kein Trigger im README) – HF: h94/IP-Adapter-FaceID |
| Klein-consistency.safetensors | (kein Trigger nötig) | – | Flux2-Klein-9B-Consistency, laut Autor ohne Cue-Wörter – HF: dx8152/Flux2-Klein-9B-Consistency |
| LCM_LoRA_Weights_SDXL.safetensors | – | – | CivitAI (kein Trigger) – LCM-LoRA Weights - Stable Diffusion Acceleration Module / LCM for SDXL (Latest) [SDXL 1.0] |
| pixel_art_style_z_image_turbo.safetensors | Pixel Art style. | – | CivitAI – Pixel Art Style Lora / v1.0 - Z Image Turbo [ZImageTurbo] |
| portrait_lora64.safetensors | – | – | HuggingFace (kein Trigger im README) – HF: artemonlysuno/comfyui_portrait_lora64 |
| schorthair.safetensors | – | – | nicht gefunden |
| Shaiden_Rogue_Flux.safetensors | – | – | CivitAI (kein Trigger) – Shaiden Rogue (Flux) - Adult Film Actress / V1 [Flux.1 D] Training-Tags: straight, well-groomed eyebrows, long |
| snofs_ideogram_v1_1.safetensors | – | – | CivitAI (kein Trigger) – Sex, Nudes, Other Fun Stuff Ideogram (SNOFS) / v1.1 [Other] |
| sweta_lora.safetensors | – | – | nicht gefunden |

## D: anna

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| m_xO4abTFgPwX3njmort3_pytorch_lora_weights.safetensors | – | – | nicht gefunden |

## D: flux

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| boFLUX Steampunk Style.safetensors | bo-steampunk / steampunk style | – | CivitAI – FLUX Steampunk Magic / FLUX v1.0 [Flux.1 D] |
| Canopus-Clothing-Flux-Dev-Florence2-LoRA.safetensors | Hoodie, Clothes, Clothing, Shirt, Pant | – | HuggingFace – HF: prithivMLmods/Canopus-Clothing-Flux-LoRA |
| Clothes_onoff.safetensors | Clothes onoff | – | CivitAI – FLUX Clothes onoff (on/off) / v2.0 [Flux.1 D] |
| collbdy.safetensors | – | – | CivitAI (kein Trigger) – Flux Bodies - Female / Colleen Body - v1 [Flux.1 D] Training-Tags: straight, this is a high-resolution, the image is a high-resolution |
| dark_fantasy_lora.safetensors | – | – | CivitAI (kein Trigger) – Flux LoRA - Dark Fantasy 80s aesthetics (popular on Tiktok) / v1.0 [Flux.1 D] |
| DollCute.safetensors | – | – | CivitAI (kein Trigger) – DollCute / V1 [Flux.1 D] Training-Tags: doll, solo, closed eyes |
| engrave.safetensors | NGRVNG | – | CivitAI – Flux Engrave LoRA / v1.0 [Flux.1 D] |
| FLUX Female Anatomy.safetensors | nude / breasts / ass / nipples / pussy | – | CivitAI – FLUX - Female Anatomy / 0.2 [Flux.1 D] |
| FLUX More Female Anatomy.safetensors | high quality / detailed / nude / naked / tits | – | CivitAI – FLUX - More Female Anatomy / v0.1 [Flux.1 D] |
| flux-fashion-lora.safetensors | null | – | HuggingFace – HF: aihpi/flux-fashion-lora |
| FLUX.1-Turbo-Alpha.safetensors | – | – | CivitAI (kein Trigger) – FLUX.1-Turbo-Alpha / FLUX.1-Turbo-Alpha [Flux.1 D] |
| FLUX_Blowjob.safetensors | Blowjob | – | CivitAI – FLUX Blowjob / v1.0 [Flux.1 D] |
| FluxK4Play.v1.safetensors | r3v3rs3_c0wg1rl / c0wg1rl / pov / m15510n4ry / d0gg13 / an4l / d0ubl3_bj | – | CivitAI – FLUX 4 PLAY - AIO NSFW FLUX Klein LoRA by DR34MSC4PE / Flux4Play_Klein9B_v1 [Flux.2 Klein 9B-base] |
| FluxPussy.safetensors | double package / close up / detailed / pussy / anus / high res / from behind / 1girl | – | CivitAI – Flux Pussy & Anus HD (From Behind) / v1.0 [Flux.1 D] |
| fluxRealSkin-V2.safetensors | realskin | – | CivitAI – Flux Skin Texture / FluxRealSkin-v2.0 [Flux.1 D] |
| fp8-lora-r640_cq1.0-fp8-flux1-fill-dev.safetensors | – | – | CivitAI (kein Trigger) – Flux.Fill - Inpaint LoRA / v1.0 [Flux.1 D] |
| Futanari_with_erect_penis_v2.safetensors | person revealing their naked genitals / nipples aren’t visible | – | CivitAI – Futanari with naked erect penis for FLUX / V1 [Flux.1 D] |
| Hyper-FLUX.1-dev-8steps-lora.safetensors | – | – | CivitAI (kein Trigger) – ByteDance Hyper-FLUX Acceleration LoRA / Hyper-FLUX.1-dev-8steps [Flux.1 D] |
| outfit-generator.safetensors | null | – | HuggingFace – HF: tryonlabs/FLUX.1-dev-LoRA-Outfit-Generator |
| Penis_on_Face_p3n1sF4ce .safetensors | p3n1sF4ce | – | CivitAI – Penis On Face Flux / Penis on face Flux v1.0 [Flux.1 D] |
| realism_lora.safetensors | – | – | CivitAI (kein Trigger) – XLabs Flux Realism LoRA / v1.0 [Flux.1 D] |
| slptp-blowjob-lora.safetensors | slptp | – | CivitAI – [FLUX LoRA] 'Sloppy Toppy' [Messy/Sloppy/Nasty Deepthroat Blowjob LoRA] [ADD: 'slptp' TO PROMPT] / v1.0 [Flux.1 D] |
| ThroatedV2_000002950.safetensors | Throating a penis | – | CivitAI – throated (flux lora testing) / v1.0 [Flux.1 D] |
| xia3.safetensors | xia | – | CivitAI – Flux Wuxia / v1.0 [Flux.1 D] |
| xlabs_flux_realism_lora_comfui.safetensors | – | – | CivitAI (kein Trigger) – XLabs Flux Realism LoRA / Converted for Comfyui [Flux.1 D] |

## D: flux2

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| bj_20260120_22-22-29epoch15_comfy.safetensors | pov / licking / cheeking / swallowing / sucking | – | CivitAI – [flux2-klein-9b] Pyro's BJ / v20260120 [Flux.2 Klein 9B] |
| gf-next-anal.safetensors | – | – | CivitAI (kein Trigger) – [FLUX.2] Real-Anatomy - Ultimate Intimacy LoRA / v1.0-alpha1 [Flux.2 Klein 9B] |

## D: flux2_klein_9b

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| [flux2-klein-9b] Pyro's BJ_v20260120.safetensors | pov / licking / cheeking / swallowing / sucking | – | CivitAI – [flux2-klein-9b] Pyro's BJ / v20260120 [Flux.2 Klein 9B] |
| [flux2-klein-9b] Pyro's Cowgirls_v1.0.safetensors | a woman sitting on a man | – | CivitAI – [flux2-klein-9b] Pyro's Cowgirls / v1.0 [Flux.2 Klein 9B] |
| bfs_head_v1_flux-klein_9b_step3500_rank128.safetensors | head_swap: start with Picture 1 as the base image, keeping its lighting, environment, and background. remove the head from Picture 1 completely and replace it… | – | CivitAI – BFS - Best Face Swap / F.K. 9b - V1 Focus Head [Flux.2 Klein 9B-base] |
| FLUX2_KLEIN_UNLOCKED_V2.safetensors | nude / naked / blow job / cum / ass / pussy | – | CivitAI – NSFW UNLOCKED [LORA] FLUX / FLUX2 KLEIN / Z_IMAGE / KREA2 / FLUX2_KLEIN_V2 [Flux.2 Klein 9B] |
| klein_9b_enhancer_v2_klein_9b_enhancer_v2.safetensors | Enhance this image | – | CivitAI – Elusarca's Detail Enhancer / Flux Klein 9B / v1.0 [Flux.2 Klein 9B] |
| V2_flux_klein_4_V2_flux_klein_4.safetensors | fluxtrait | – | CivitAI – Portrait Engine FLUX / FLUX 2 KLEIN / Z-Image, Detailed Skin - [LORA] / V 2.0_Flux_klein [Flux.2 Klein 9B] |

## D: flux\AWPortraitFL

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| AWPortrait-FL-lora.safetensors | – | – | nicht gefunden – Training-Tags: realistic photo, photo, photography |

## D: Krea2

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| depth-control-lora.safetensors | – | – | CivitAI (kein Trigger) – Krea 2 Depth ControlNet LoRA / v1.0 [Krea 2] |
| KNP_V2_copy_copy.safetensors | – | – | CivitAI (kein Trigger) – Krea 2 NSFW V4 / V2 [Krea 2] Training-Tags: KNP |
| Krea2-depth-control-lora.safetensors | – | – | CivitAI (kein Trigger) – Krea 2 Depth ControlNet LoRA / v1.0 [Krea 2] |
| krea2_turbo_distill_r64.safetensors | – | – | CivitAI (kein Trigger) – Krea-2 Turbo 8-Step Distillation LoRA (SVD Extract) / V2026.1 [Krea 2] |
| lenovo_krea2.safetensors | – | – | CivitAI (kein Trigger) – Lenovo UltraReal / v1.0 Krea2 [Krea 2] |
| Purple_Grainy_Kr2_AM.safetensors | – | – | CivitAI (kein Trigger) – Purple Grainy // Photography lora / v1.0 Krea 2 [Krea 2] Training-Tags: |
| realism_engine_krea2_v2.safetensors | – | – | CivitAI (kein Trigger) – Realism Engine Ideogram 4 + Krea 2 / Krea2 v2.0 [Krea 2] |

## D: minimax

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| H3_Motion_BoosterV2.safetensors | dynv2 | – | CivitAI – Faster! Harder! Shake Harder! / H3 Motion Booster(ANIME Edition Update) / V0.2 [MiniMax H3] |
| Minimax_H3_Unlocked_V2.safetensors | – | – | CivitAI (kein Trigger) – MiniMax H3 NSFW Unlocked / Minimax_H3_V2 [MiniMax H3] |
| TaoMate-H3-3step.safetensors | – | – | CivitAI (kein Trigger) – TaoMate-H3-3step / v1.0 [MiniMax H3] |

## D: qwen

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| bfs_face_v1_qwen_image_edit_2509.safetensors | face swap face from Image 1 to Image 2. swap only the face and not the hair, the same skin tone from Image 2, same pose that Image 3 / face swap face from Imag… | – | CivitAI – BFS - Best Face Swap / QIE 2509 - V1 Focus Face [Qwen] |
| DiffSynth-StudioQwenImage2.1LayerExtract.safetensors | – | – | nicht gefunden |
| DiffSynthStudioQwenImage2.1LayerRemove.safetensors | – | – | nicht gefunden |
| NSFW-Qwen-by-TheseAlpacas-V2.safetensors | – | – | CivitAI (kein Trigger) – NSFW LORA / Qwen Image 2.1 / v2.0 [Qwen 2.1] Training-Tags: nsfw_qwen_lora |
| p_qwen_image_2.1_5step_v0.1.safetensors | – | – | CivitAI (kein Trigger) – Turbo - Few-step LoRA adapters - Pruna Qwen Image 2.1 / pruna 5 steps [Qwen 2.1] |
| p_qwen_image_2.1_8step_v0.1.safetensors | – | – | CivitAI (kein Trigger) – Turbo - Few-step LoRA adapters - Pruna Qwen Image 2.1 / pruna 8 steps [Qwen 2.1] |
| Qwen-Image-2.1-Fun-Acc-4Step.safetensors | – | – | nicht gefunden |
| Qwen-Image-Edit-2511-Lightning-4steps-V1.0-bf16.safetensors | – | – | CivitAI (kein Trigger) – Qwen Image Edit Lightning / 4steps bf16 (2511) [Qwen] |
| qwen-image-edit-2511-multiple-angles-lora.safetensors | – | – | CivitAI (kein Trigger) – Qwen-Image-Edit-2511-Multiple-Angles-LoRA / v1.0 [Qwen] |
| Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors | – | – | HuggingFace (kein Trigger im README) – HF: lightx2v/Qwen-Image-Lightning |
| qwen-image_nsfw_adv_v1.0.safetensors | – | – | CivitAI (kein Trigger) – Qwen-image_NSFW_Adv.1 / v1.0 [Qwen] |
| qwen2.1_penisV01_000004956.safetensors | – | – | CivitAI (kein Trigger) – Qwen image 2.1 Perfect erect penis / v1.0 [Qwen 2.1] |

## D: qwen\VNCCS

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| VNCCS_QIE2511_PoseStudio_ART_V5.9.safetensors | – | – | nicht gefunden |

## D: wan

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| DR34ML4Y_I2V_14B_HIGH.safetensors | m15510n4ry / bl0wj0b / d0ubl3_bj / d0gg1e / c0wg1rl | – | CivitAI – DR34ML4Y - All-In-One NSFW (WAN/LTX2) / DR34ML4Yv1.0_I2V_14B_HIGH [Wan Video 2.2 I2V-A14B] |
| DR34ML4Y_I2V_14B_LOW.safetensors | m15510n4ry / bl0wj0b / d0ubl3_bj / d0gg1e / c0wg1rl | – | CivitAI – DR34ML4Y - All-In-One NSFW (WAN/LTX2) / DR34ML4Y_v1.0_I2V_14B_LOW [Wan Video 2.2 I2V-A14B] |
| DR34ML4Y_LTXXX_V2.safetensors | m15510n4ry / bl0wj0b / d0ubl3_bj / d0gg1e / c0wg1rl / r3v3rs3_c0wg1rl | – | CivitAI – DR34ML4Y - All-In-One NSFW (WAN/LTX2) / DR34ML4Y_LT3X_v2 [LTXV 2.3] |
| Wan21_I2V_14B_lightx2v_cfg_step_distill_lora_rank64.safetensors | – | – | CivitAI (kein Trigger) – Lightning Lora, massive speed up for Wan2.1 / Wan2.2 made by Lightx2v / Kijai / Self-Forcing 14B I2V [Wan Video 14B i2v 480p] |
| Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors | – | – | HuggingFace (kein Trigger im README) – HF: estaksi/Wan21_PusaV1_LoRA_14B_rank512_bf16.safetensors |

## D: z_image

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| NSFW_ZI_V01_r1_Epoch31_AstroNSFW-V02-ZIG.safetensors | – | – | CivitAI (kein Trigger) – Z-Image Turbo NSFW / V2-Released [ZImageTurbo] Training-Tags: astronsfw-v02-zig, the image is a high-resolution, straight |

## Aus der alten Liste (liegen in latent_upscale_models, nicht im LoRA-Ordner)

| Dateiname | Aktivierungswort | Stärke | Quelle / Info |
|---|---|---|---|
| ltx-2.3-spatial-upscaler-x2-1.1.safetensors | – | 1.0 | loras.md (alt), Latent-Upscaler, kein Trigger |
| ltx-2.3-temporal-upscaler-x2-1.0.safetensors | – | 1.0 | loras.md (alt), Latent-Upscaler, kein Trigger |
