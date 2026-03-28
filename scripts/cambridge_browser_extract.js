/**
 * Cambridge Multi-track Library - 브라우저 콘솔 추출 스크립트
 *
 * 사용법:
 *   1. 브라우저에서 https://cambridge-mt.com/ms3/mtk/ 열기
 *   2. F12 → Console 탭
 *   3. 아래 전체 복사 → 붙여넣기 → Enter
 *   4. cambridge_mt_data.json 파일이 자동 다운로드됨
 */
(function () {
  const songs = [];

  // 각 곡 항목 파싱 (테이블 행 or 섹션 기반)
  const rows = document.querySelectorAll('table tr, .mtk-entry, .song-entry');

  rows.forEach((row) => {
    const text = row.innerText || '';
    const links = Array.from(row.querySelectorAll('a[href]'));

    // 다운로드 링크 추출 (.zip 포함)
    const zipLinks = links
      .filter(a => a.href.includes('.zip') || a.href.includes('download'))
      .map(a => ({ text: a.innerText.trim(), url: a.href }));

    // 악기 정보 추출 (텍스트에서 키워드 찾기)
    const lower = text.toLowerCase();
    const hasAcousticGuitar = /acoustic\s*guitar|gac|acoustic gtr/.test(lower);
    const hasElectricGuitar = /electric\s*guitar|gel|electric gtr|elec\s*gtr/.test(lower);
    const hasBoth = hasAcousticGuitar && hasElectricGuitar;

    // 곡 제목 (첫 번째 링크 텍스트 or 행의 첫 셀)
    const titleEl = row.querySelector('td:first-child, h3, h4, .title');
    const title = titleEl ? titleEl.innerText.trim() : text.split('\n')[0].trim();

    if (zipLinks.length > 0 || hasBoth) {
      songs.push({
        title,
        hasAcousticGuitar,
        hasElectricGuitar,
        hasBoth,
        zipLinks,
        rawText: text.substring(0, 300),
      });
    }
  });

  // 결과 요약
  const bothCount = songs.filter(s => s.hasBoth).length;
  console.log(`총 ${songs.length}개 항목 발견`);
  console.log(`통기타 + 일렉 둘 다 있는 곡: ${bothCount}개`);

  // JSON 파일 자동 다운로드
  const blob = new Blob([JSON.stringify(songs, null, 2)], { type: 'application/json' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'cambridge_mt_data.json';
  a.click();

  console.log('cambridge_mt_data.json 다운로드 완료!');
  console.log('이 파일을 music-assistant/ 폴더에 넣고 cambridge_downloader.py 실행하세요.');

  return songs;
})();
