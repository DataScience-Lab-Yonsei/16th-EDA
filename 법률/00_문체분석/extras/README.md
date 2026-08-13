# 참고용 산출물 생성 코드

이 폴더의 파일은 핵심 통계 분석과 분리된 참고 자료입니다.

- `build_integrated_report.py`: 분석 표·그림을 Word 통합보고서로 조립
- `render_with_word.ps1`: Microsoft Word를 이용해 DOCX를 PDF로 변환
- `build_deck.mjs`: 분석 표·그림을 PowerPoint로 조립

주의:

- 작성 당시의 Codex 번들 도구 또는 로컬 Microsoft Word 환경에 의존합니다.
- 일부 경로는 작업 당시의 로컬 경로이므로 다른 PC에서 실행하려면 수정해야 합니다.
- 분석 결과를 재현하는 데에는 `src/01`부터 `src/09`까지만 필요합니다.
