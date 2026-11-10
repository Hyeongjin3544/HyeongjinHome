# Hyeongjin Kim · Personal Academic Homepage

경제학 대학원생 **김형진 (Hyeongjin Kim)**의 개인 프로필, 연구 논문, 강의 경력, 실증 프로젝트 및 스터디 노트를 담은 현대적 아카데믹 웹사이트입니다.

🌐 **웹사이트 주소:** [https://hyeongjin3544.github.io/HyeongjinHome/](https://hyeongjin3544.github.io/HyeongjinHome/)

---

## 🌟 주요 특징

- **초고속 & 100% 무에러 배포:** 복잡한 빌드 툴 없이 브라우저에서 바로 열리는 순수 정적 웹사이트 (Tailwind CSS, KaTeX 수식, FontAwesome 지원)
- **모던 아카데믹 디자인:** 다크 모드 / 라이트 모드 지원, Playfair Display 세리프 폰트 및 감각적인 슬레이트 & 딥블루 톤
- **경제학 연구자 특화 섹션:**
  - **About:** 연구 비전 및 소개
  - **Research:** Working Papers, Works in Progress, Publications (초록 아코디언 토글, PDF/Code/Slides 링크)
  - **Teaching:** 대학원/학부 조교(TA) 경험 및 자료 링크
  - **Projects:** R, Python, Stata 기반 실증 데이터 분석 쇼케이스
  - **Study Notes:** $\LaTeX$ 수식이 지원되는 개별 기술 노트 및 페이퍼 리뷰
  - **Curriculum Vitae (CV):** 학술 이력서 및 PDF 다운로드 버튼
  - **Contact:** 오피스 위치, 오피스 아워, 이메일

---

## 📁 파일 구조

```text
├── index.html                  # 메인 전체 포털 (반응형 싱글 페이지 + 네비게이션)
├── .nojekyll                   # GitHub Pages 초고속 직통 배포 보장 파일
├── notes/                      # 개별 상세 연구 노트 페이지
│   ├── causal-inference-did.html
│   └── linear-market.html
├── assets/                     # 프로필 이미지, 다이어그램
│   ├── avatar.svg
│   └── market-equilibrium.svg
├── .github/workflows/          # GitHub Actions 무오류 정적 배포 워크플로우
│   └── deploy.yml
└── README.md
```

---

## ✏️ 내 정보로 쉽게 수정하기

1. `index.html` 파일을 텍스트 에디터(VS Code, 메모장 등)로 열어 본인의 이메일(`your-email@example.com`), 소속 대학교 이름, 세부 연구 소개를 수정합니다.
2. `assets/avatar.svg` 대신 본인의 실제 사진을 `assets/profile.jpg`로 저장하고 `index.html`의 `src="assets/avatar.svg"`를 `src="assets/profile.jpg"`로 변경합니다.
3. 로컬에서 `index.html` 파일을 더블클릭하면 브라우저에서 즉시 수정된 모습을 확인하실 수 있습니다.
