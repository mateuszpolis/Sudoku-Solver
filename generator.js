const { generateSudoku } = require("sudoku-puzzle");
const fs = require('fs');

let puzzles = [];
for (let i = 0; i < 50000; i++) {
  const puzzle = generateSudoku(9, 5);
  puzzles.push(puzzle);
}

let output = `${puzzles.length}\n\n`;
puzzles.forEach(puzzle => {
  puzzle.forEach(row => {
    output += row.join('') + '\n';
  });
  output += '\n';
});

fs.writeFileSync('boards.txt', output);
