# Vídeo: Box Hyrox no Jardim Itu (montagem na nuvem)

Montado com HyperFrames 0.8.142 a partir do `ROTEIRO.md` da sessão anterior.

## O que tem aqui
- `projeto/index.html`: a composição (11 cenas, ~2min15s, 1920x1080, 30 fps).
- `projeto/assets/narracao.wav`: narração completa, voz Kokoro `pm_santa` (a mesma das amostras do Voicebox).
- `projeto/assets/cena-*.mp4`: **clipes provisórios** (fundo listrado com o nome do arquivo).
  Os clipes reais do Veo estão no PC, em `box-hyrox-video/assets/`.
- `voz/trechos.txt`: o texto de cada trecho, um por linha. `voz/gerar_trechos.py <voz>` gera os wavs.

## Para trocar pelos clipes reais
Copie os 5 arquivos com os mesmos nomes para `projeto/assets/` e renderize de novo:
`cena-treno.mp4`, `cena-ski.mp4`, `cena-turma.mp4`, `cena-loja.mp4`, `cena-corrida.mp4`.
Os vídeos entram sem som (`muted`).
O cartaz com letras embaralhadas em `cena-turma.mp4` fica sob um escurecimento de 72%.

```bash
cd projeto
npx hyperframes@0.8.142 check
npx hyperframes@0.8.142 render -o ../box-hyrox.mp4
```

## Para trocar a voz
```bash
python voz/gerar_trechos.py pf_dora   # ou pm_alex
```
Depois, monte de novo `projeto/assets/narracao.wav`. Se a duração de algum trecho mudar,
os tempos das cenas (`data-start` e as listas `S` e `D` no script) precisam acompanhar.
