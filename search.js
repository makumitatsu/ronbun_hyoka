import { pipeline, env } from './transformers.js';

// --- グローバル設定 ---
env.allowLocalModels = false;

// AIモデルを非同期で準備
const extractor_promise = (async () => {
    const statusDiv = document.getElementById('results');
    statusDiv.innerHTML = '<p class="loading-message">初回起動のため、検索用AIモデルをインターネットからダウンロードしています...（数十秒～数分かかる場合があります）</p>';
    const extractor = await pipeline('feature-extraction', 'Xenova/multilingual-e5-base');
    statusDiv.innerHTML = '<p class="loading-message">モデル準備完了。検索したい内容を入力してください。</p>';
    return extractor;
})();


// --- 共通ヘルパー関数 ---
function cosineSimilarity(vecA, vecB) {
    if (!vecA || !vecB) return 0; // 安全策
    let dotProduct = 0, normA = 0, normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}


// --- Mode 1: セマンティック検索機能 ---
const searchButton = document.getElementById('searchButton');
searchButton.addEventListener('click', searchPapers);

async function searchPapers() {
    const resultsDiv = document.getElementById('results');
    resultsDiv.innerHTML = '<p class="loading-message">検索中...</p>';
    try {
        const [response, extractor] = await Promise.all([fetch('papers_with_eval.json'), extractor_promise]);
        const papers = await response.json();
        const query = document.getElementById('searchInput').value;
        if (!query) {
            resultsDiv.innerHTML = '<p class="error-message">検索語を入力してください。</p>';
            return;
        }
        const queryEmbedding = await extractor('query: ' + query, { pooling: 'mean', normalize: true });
        papers.forEach(paper => {
            paper.score = cosineSimilarity(queryEmbedding.data, paper.embedding);
        });
        const topResults = papers.sort((a, b) => b.score - a.score).slice(0, 3);
        if (topResults.length === 0 || topResults[0].score < 0.6) {
            resultsDiv.innerHTML = '<p class="error-message">関連する論文は見つかりませんでした。</p>';
            return;
        }
        resultsDiv.innerHTML = '<h3>関連性の高い論文:</h3>';
        topResults.forEach(paper => {
            resultsDiv.innerHTML += `<div class="paper-card"><h4>${paper.filename}</h4><p><strong>内容のプレビュー:</strong> ${paper.content}</p><p><strong>関連度スコア:</strong> ${paper.score.toFixed(3)}</p></div>`;
        });
    } catch (error) {
        console.error("検索中にエラーが発生しました:", error);
        resultsDiv.innerHTML = '<p class="error-message">検索中にエラーが発生しました。</p>';
    }
}


// --- Mode 2: 論文評価・比較機能 ---
const basePaperSelect = document.getElementById('basePaperSelect');
const compareButton = document.getElementById('compareButton');
const mainChartCanvas = document.getElementById('mainRadarChart'); // 総合チャート用
const detailsContainer = document.getElementById('comparison-details-container'); // 個別比較用

let allPapersEvalData = [];
let allComparisonsData = {};
let createdCharts = []; // 作成した全チャートのインスタンスを保持

async function initializeMode2() {
    try {
        const [evalResponse, comparisonsResponse] = await Promise.all([
            fetch('papers_with_eval.json'),
            fetch('final_comparison_data.json')
        ]);
        allPapersEvalData = await evalResponse.json();
        allComparisonsData = await comparisonsResponse.json();
        allPapersEvalData.forEach(paper => {
            const option = new Option(paper.filename, paper.filename);
            basePaperSelect.add(option);
        });
    } catch (error) {
        console.error("データファイルの読み込みに失敗:", error);
    }
}

compareButton.addEventListener('click', () => {
    createdCharts.forEach(chart => chart.destroy());
    createdCharts = [];
    detailsContainer.innerHTML = "";

    const baseFilename = basePaperSelect.value;
    const comparisons = allComparisonsData[baseFilename];
    if (!comparisons || comparisons.length < 3) {
        detailsContainer.innerHTML = "<p class='error-message'>この論文の比較データが見つかりません。</p>";
        return;
    }

    const basePaper = allPapersEvalData.find(p => p.filename === baseFilename);
    const top3SimilarPapers = comparisons.map(comp => allPapersEvalData.find(p => p.filename === comp.compared_to));

    // 1. 最初に総合チャート（4データ）を描画
    drawRadarChart(mainChartCanvas, [basePaper, ...top3SimilarPapers]);

    // 2. 次に個別比較ブロック（グラフ＋コメント）を3つ生成
    comparisons.forEach((comp, index) => {
        const relatedPaper = top3SimilarPapers[index];
        if (basePaper && relatedPaper) {
            renderComparisonBlock(basePaper, relatedPaper, comp, index);
        }
    });
});

function renderComparisonBlock(basePaper, relatedPaper, comparisonData, index) {
    const blockContainer = document.createElement('div');
    blockContainer.className = 'comparison-block';
    const analysis = comparisonData.analysis;
    
    // ▼▼▼ コメント表示を8項目に増やす ▼▼▼
    const commentHtml = analysis ? `
        <dl>
            <dt><strong>サマリー:</strong></dt><dd>${analysis.comparison_summary}</dd>
            <dt><strong>手法の比較:</strong></dt><dd>${analysis.method_comparison}</dd>
            <dt><strong>結果の比較:</strong></dt><dd>${analysis.result_comparison}</dd>
            <dt><strong>進化・変化の考察:</strong></dt><dd>${analysis.evolution_analysis}</dd>
        </dl>
    ` : `<p class="error-message">分析コメントデータがありません。</p>`;

    blockContainer.innerHTML = `
        <div class="paper-card">
            <h3>「${basePaper.filename.replace('.docx','')}」 vs 「${relatedPaper.filename.replace('.docx','')}」</h3>
            <p><strong>関連度スコア:</strong> ${comparisonData.similarity_score.toFixed(3)}</p>
            <div class="chart-container" style="max-width: 600px; margin: auto;">
                <canvas id="radarChart-individual-${index}"></canvas>
            </div>
            <hr>
            <h4>比較分析コメント</h4>
            ${commentHtml}
        </div>
    `;
    detailsContainer.appendChild(blockContainer);

    const individualCanvas = document.getElementById(`radarChart-individual-${index}`);
    drawRadarChart(individualCanvas, [basePaper, relatedPaper]);
}

function drawRadarChart(canvasElement, papersToCompare) {
    // ...
    // ▼▼▼ ラベルを8項目に変更 ▼▼▼
    const labels = ['新規性', '解析の質', 'わかりやすさ', '実験妥当性', '貢献度', '文章構成', '網羅性', '考察の深さ'];
    const colors = ['rgba(255, 99, 132, 1)', 'rgba(54, 162, 235, 1)', 'rgba(75, 192, 192, 1)', 'rgba(255, 206, 86, 1)'];
    
    const datasets = papersToCompare.map((paper, index) => {
        const evalScores = paper.evaluation;
        const getData = (prop) => evalScores?.[prop]?.score ?? 0;
        return {
            label: paper.filename.replace('.docx', ''),
            // ▼▼▼ データを8項目に増やす ▼▼▼
            data: [
                getData('novelty'),
                getData('analysis_quality'),
                getData('clarity'),
                getData('experiment_validity'),
                getData('contribution'),
                getData('structure'),
                getData('comprehensiveness'), // 追加
                getData('depth_of_discussion')  // 追加
            ],
            borderColor: colors[index % colors.length],
            backgroundColor: colors[index % colors.length].replace('1)', '0.2)'),
            pointBackgroundColor: colors[index % colors.length]
        };
    });

    const newChart = new Chart(canvasElement, {
        type: 'radar',
        data: { labels: labels, datasets: datasets },
        options: {
            responsive: true,
            maintainAspectRatio: true,
            elements: { line: { borderWidth: 3 } },
            scales: { r: { angleLines: { display: true }, suggestedMin: 0, suggestedMax: 5, pointLabels: { font: { size: 14 } } } }
        }
    });
    createdCharts.push(newChart);
}

document.addEventListener('DOMContentLoaded', initializeMode2);