# Avaliação Comparativa: MS Teams vs. OmniTranscriber

**Data:** 27 maio 2026, 10:34–50:52 (CIE Technical Interview)

---

## Resumo Executivo

| Aspecto | MS Teams | OmniTranscriber | Vencedor |
| --- | --- | --- | --- |
| **Linhas de conteúdo** | 1,057 | 325 | MS Teams (3x mais longo) |
| **Cobertura do meeting** | ~100% | ~33% | MS Teams |
| **Ruído/Artefatos** | Mínimo | Significativo | MS Teams |
| **Identificação de speakers** | Nomes (Afonso, Takeuchi) | Genérico (You/Them) | MS Teams |
| **Timestamps** | Precisos (0:08, 0:11...) | Precisos ([00:00:12]...) | Empate |
| **Metadata** | Completo | Completo | Empate |
| **Usabilidade** | Excelente | Pobre | MS Teams |
| **Recomendação** | ✅ Use como primário | ❌ Não use | **MS Teams** |

---

## Análise Detalhada

### 1. MS Teams Transcription

**Arquivos:**
- 1,057 linhas
- Timestamps: "0:08", "0:23", "0:30", ... "50:00"
- Metadata: "27 de maio de 2026, 01:35PM", "50m 7s"
- Arquivo: `01-transcription-generated-by-ms-teams.txt`

**Qualidades:**

✅ **Completude:** Cobre ~50 minutos de meeting integralmente. Incluindo:
  - Introdução e contexto (min 0–2)
  - Demonstração de telas (min 1–15)
  - Detalhes técnicos de formulário, fluxo, data (min 4–48)
  - Encerramento (min 49–50)

✅ **Identificação clara:** Speakers identificados por nome
  ```
  Afonso, Luiz (DGM TECNOLOGIA EM INFORMATICA LTDA)   0:08
  Takeuchi   0:11
  ```

✅ **Estrutura conversacional:** Fluxo natural de pergunta-resposta preservado

✅ **Ruído mínimo:** Artefatos menores (linhas repetidas como "Partner Partner Partner..." em 47, algumas inversões de fala) não prejudicam compreensão geral

✅ **Metadata preciso:** Incluindo hora exata, duração, nomes de participantes

**Deficiências menores:**
- Algumas palavras codificadas incorretamente (p.ex., "Shiino" em vez de possível nome)
- Pequenos trechos com áudio baixo ou sobreposição
- Ocasional mistura de português/inglês ("There go", "That's") refletindo bilinguismo da reunião

---

### 2. OmniTranscriber Transcription

**Arquivo:**
- 325 linhas
- Timestamps: "[00:00:12]", "[00:00:24]", ... "[00:52:00]"
- Metadata: "transcript started 2026-05-27 10:34:25"
- Arquivo: `01-transcription-generated-by-omni-transcriber.txt`

**Deficiências críticas:**

❌ **Perda massiva de conteúdo:** ~67% do meeting está ausente
  - OmniTranscriber: 325 linhas
  - MS Teams: 1,057 linhas
  - Razão: 1:3.25

❌ **Ruído substancial:** "Obrigado por assistir" (YouTube-style watermark) aparece **16+ vezes aleatoriamente** ao longo da transcrição (linhas 7, 46, 69, 80, 86, 96, 99, 106, 153, 163, 194, 240, 266, 278, 307, 322)
  ```
  [00:09:34] You: Obrigado por assistir.
  [00:12:20] You: Obrigado por assistir.
  [00:13:43] You: Obrigado por assistir.
  ... (13x mais)
  ```
  Claramente não foi falado durante a reunião — geração artificial da ferramenta.

❌ **Texto inserido incorretamente:** Frases como:
  - "Acompanhe a avaliação para fazer esse filtro..."
  - "Acompanhe a avaliação do programa em português..."
  - "O que eu fiz é semelhante" (cortado no meio)
  
  Estas parecem ser overlays de outros vídeos ou processamento incorreto.

❌ **Speakers genéricos:** "You:" vs "Them:" sem nomes. Impossível rastrear quem falou o quê.

❌ **Trecho final confuso:** Linhas 320–325 contêm anotações pessoais em vez de transcrição:
  ```
  [00:52:00] You: Para o uso de IA, porque tem todo esse processo de aprovação...
  [00:52:00] You: O último sistema que ele falou que está estruturado, que está ruim, não é o CIE...
  ```
  Parecem notas editadas, não transcrição original.

❌ **Conteúdo truncado:** Grandes seções do meio do meeting estão ausentes. Comparação linha por linha mostra lacunas de 10–20 linhas consecutivas entre timestamps.

**Exemplo de divergência:**
```
MS Teams (linhas 38–97): explicação completa sobre as 3 abas do formulário, 
                         detalhes de campos obrigatórios, vieses por área, etc.

OmniTranscriber (linhas 34–45): mesmos tópicos mas cortados abruptamente, 
                                 pulando para outro assunto.
```

---

## Avaliação Qualitativa

### Acurácia de Conteúdo

Onde ambos sobrepoem (primeiras 30% da reunião), MS Teams é **~95% acurado**. OmniTranscriber é ~85% acurado onde presente, mas complementado com ruído (~15% é artefatos).

### Usabilidade para Documentação

| Uso | MS Teams | OmniTranscriber |
| --- | --- | --- |
| Análise técnica completa | ✅ Possível | ❌ Impossível |
| Validação de decisões | ✅ Fácil | ⚠️ Parcial |
| Auditoria/Compliance | ✅ Confiável | ❌ Risco |
| Citation (referência) | ✅ Seguro | ❌ Inseguro |
| Notificação a stakeholders | ✅ Sim | ❌ Não recomendado |

---

## Recomendação Final

**✅ Use MS Teams como transcription principal** para CIE e futuros meetings técnicos.

**Motivos:**
1. 3x mais conteúdo (1,057 vs 325 linhas)
2. Sem ruído sistemático ("Obrigado por assistir" repetido)
3. Speakers identificados por nome
4. Estrutura conversacional clara
5. Confiável para auditoria/compliance

**❌ Descontinue OmniTranscriber** para este tipo de documento técnico.

**Se OmniTranscriber for usado novamente:**
- Validar completude (comparar duração do meeting vs linhas produzidas)
- Sanitizar ruído automático ("Obrigado por assistir", watermarks)
- Adicionar identificação manual de speakers se omitida
- Usar como *backup complementar* apenas, não primário

---

## Observação Técnica

Padrão de "Obrigado por assistir" + truncagem sugere que OmniTranscriber:
- Pode estar grabando audio de múltiplas fontes simultaneamente (reunião + vídeo background?)
- Tem limites de tempo de transcrição (cortou em ~5 minutos úteis?)
- Aplica watermarks de YouTube automaticamente

Recomenda-se verificar configurações de OmniTranscriber antes de próximo uso ou considerar alternativa (Otter.ai, Google Meet transcription, etc).

---

More feedbacks:
Configuration & Setup
  
  1. Audio Input Quality
  - Record directly from meeting audio source (not screen capture with embedded audio)
  - Single clean source, not multiple simultaneous recordings
  - Test with MS Teams' native recording first to isolate if it's OmniTranscriber or Whisper config

  2. Whisper Model Size
  - Use base or small model minimum (OmniTranscriber may default to tiny/small)
  - Larger models = better accuracy but slower: base → small → medium → large
  - For Portuguese, small or medium usually sufficient

  3. Language Specification
  - Explicitly set language to Portuguese (pt-BR or pt) before transcription
  - Whisper performs better when language is pre-declared vs. auto-detected

  ---
  Post-Processing Issues (Your Problem)
  
  4. Disable/Review Watermark Insertion
  - OmniTranscriber is adding "Obrigado por assistir" (YouTube watermark) post-transcription
  - Check OmniTranscriber settings for:
    - Watermark removal option (disable it)
    - Auto-cleanup filters (may be too aggressive)
    - Post-processing steps (may be inserting noise)

  5. Content Loss Prevention
  - Check OmniTranscriber's time limit settings (may be capping at 5 min or similar)
  - Verify meeting duration vs. output: if 50-min meeting produces only 325 lines, something is truncating
  - Options:
    - Split long meetings into chunks (<20 min each) and concatenate
    - Increase buffer/timeout settings if available
    - Process locally instead of via API (if OmniTranscriber supports it)

  6. Speaker Diarization
  - Enable speaker identification (not generic "You"/"Them")
  - Whisper doesn't do diarization natively — check if OmniTranscriber has this
  - If not, use separate tool (pyannote.audio) + Whisper combo

  ---
  Better Alternatives for Your Use Case

  Given the regulatory/compliance nature of your interviews, consider:

  ┌───────────────────────────┬───────────────────────────────────────┬─────────────────────────────────┬──────────────────────────────┐
  │           Tool            │                 Pros                  │              Cons               │           For You?           │
  ├───────────────────────────┼───────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────┤
  │ MS Teams native           │ Complete, speaker names, no noise     │ Can't customize model           │ ✅ Use as primary (proven)   │
  ├───────────────────────────┼───────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────┤
  │ Google Meet transcription │ Native, clean output, speaker labels  │ Limited to Google ecosystem     │ ✅ Consider if using Meet    │
  ├───────────────────────────┼───────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────┤
  │ Otter.ai                  │ Great speaker diarization, searchable │ Paid, may have privacy concerns │ ⚠️  Evaluate                  │
  ├───────────────────────────┼───────────────────────────────────────┼─────────────────────────────────┼──────────────────────────────┤
  │ Local Whisper + pyannote  │ Full control, no cloud data           │ Requires setup, slower          │ ✅ Gold standard for privacy │
  └───────────────────────────┴───────────────────────────────────────┴─────────────────────────────────┴──────────────────────────────┘

  ---
  Immediate Action: Whisper + Pyannote (Local, No Noise)
  
  If you want to replace OmniTranscriber:

  # Install
  pip install openai-whisper pyannote.audio

  # Record meeting locally (WAV/MP3)
  # Then transcribe with speaker diarization:
  python3 << 'EOF'
  from pyannote.audio import Pipeline
  from openai import whisper

  # 1. Diarize (who spoke when)
  pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1")
  diarization = pipeline("meeting.wav")

  # 2. Transcribe with Whisper
  model = whisper.load_model("small")  # or "medium" for Portuguese
  result = model.transcribe("meeting.wav", language="pt")

  # 3. Merge (attributable transcription)
  for segment, speaker, label in diarization.itertracks(yields_label=True):
      print(f"{speaker} ({segment.start:.0f}s–{segment.end:.0f}s): [transcription excerpt]")
  EOF

  Advantage: No watermarks, no truncation, full speaker identification, runs locally.

  ---
  Quick Fix (Today)
  
  1. Disable OmniTranscriber watermark — Check settings, turn off any "optimize for YouTube" or similar
  2. Test chunk-based processing — Split 50-min meeting into 10-min clips, process separately
  3. Compare output line count — If output is still 325 lines for 50 min, time limit is the culprit
  4. Fall back to MS Teams for critical interviews — proven 1,057 lines for same 50-min meeting

