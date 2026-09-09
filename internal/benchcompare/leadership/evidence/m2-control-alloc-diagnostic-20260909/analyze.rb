# Diagnostic-only exact-integer protocol audit; no qualification verdict.
# Usage: ruby analyze.rb OLD_OLD OLD_V2 OLD_SHA256 V2_SHA256 > analysis.csv
require 'json'
require 'csv'

N = 1024
NAMES = %w[BenchmarkSiLUBackwardF64_256K_cpu BenchmarkSigmoidF64_64K_cpu BenchmarkSoftplusF64_256K_cpu].freeze
INTEGER_FIELDS = %w[procs n total_ns total_bytes total_allocs bytes_per_op bytes_remainder allocs_per_op allocs_remainder].freeze
FIELDS = (['benchmark'] + INTEGER_FIELDS).sort.freeze

class UniqueObject < Hash
  def []=(key, value)
    raise "Duplicate JSON key #{key}" if key?(key)
    super
  end
end

def median(values)
  values.sort.fetch(values.length / 2)
end

def rank_p(a, b)
  all = a + b
  return 1.0 if all.uniq.length == 1
  ranks = all.map { |v| all.count { |x| x < v } + (all.count(v) + 1) / 2.0 }
  center = 52.5
  distance = (ranks.first(7).sum - center).abs
  count = (0...14).to_a.combination(7).count do |indexes|
    (indexes.sum { |i| ranks[i] } - center).abs >= distance - 1e-12
  end
  count / 3432.0
end

def parse(path, phase, old_sha, v2_sha)
  header, meta, batches = {}, {}, []
  File.foreach(path) do |line|
    line = line.strip
    if line =~ /\A(protocol|phase|old-sha256|v2-sha256|old-path|v2-path): (.+)\z/
      key, value = Regexp.last_match.captures
      raise "Duplicate/late header #{key}" if header.key?(key) || !batches.empty?
      header[key] = value
    elsif line =~ /\A(campaign|pair|procs): (\d+)\z/
      key, value = Regexp.last_match.captures
      raise 'Metadata inside incomplete invocation' unless batches.empty? || batches.last['pass']
      raise 'Duplicate invocation metadata' if meta.key?(key)
      meta[key] = value.to_i
    elsif line =~ /\Aarm: ([AB])\z/
      raise 'Missing or unordered invocation metadata' unless meta.keys == %w[campaign pair procs]
      batches << meta.merge('arm' => Regexp.last_match(1), 'rows' => [], 'pass' => false)
      meta = {}
    elsif line =~ /\Abinary-sha256: ([0-9a-f]{64})\z/
      raise 'Unexpected/duplicate binary hash' if batches.empty? || batches.last.key?('hash')
      batches.last['hash'] = Regexp.last_match(1)
    elsif line.start_with?('allocdiag: ')
      raise 'Record outside invocation' if batches.empty? || batches.last['pass']
      row = JSON.parse(line.delete_prefix('allocdiag: '), object_class: UniqueObject)
      raise 'Unexpected JSON fields' unless row.is_a?(Hash) && row.keys.sort == FIELDS
      raise 'Non-integer or negative metric' unless INTEGER_FIELDS.all? { |k| row[k].is_a?(Integer) && row[k] >= 0 }
      raise 'Wrong count/process or nonpositive elapsed time' unless row['n'] == N && row['procs'] == batches.last['procs'] &&
        row['total_ns'] > 0
      %w[bytes allocs].each do |metric|
        raise "Invalid #{metric} arithmetic" unless row["total_#{metric}"].divmod(N) ==
          row.values_at("#{metric}_per_op", "#{metric}_remainder")
      end
      batches.last['rows'] << row
    elsif line == 'PASS'
      raise 'Unexpected/duplicate PASS' if batches.empty? || batches.last['pass']
      batches.last['pass'] = true
    elsif line.include?('FAIL') || line.include?('SKIP') || line.start_with?('panic:', 'fatal error:')
      raise "Invalid invocation: #{line}"
    end
  end
  raise 'Trailing incomplete metadata' unless meta.empty?
  raise 'Wrong protocol/phase/frozen hashes' unless header['protocol'] == 'control-alloc-fixed1024-v1' &&
    header['phase'] == phase && header['old-sha256'] == old_sha && header['v2-sha256'] == v2_sha
  raise 'Missing absolute paths' unless %w[old-path v2-path].all? { |k| header.fetch(k, '').start_with?('/') }
  expected = (1..3).flat_map do |campaign|
    (1..7).flat_map do |pair|
      (campaign.odd? ? [1, 12] : [12, 1]).flat_map do |procs|
        ((pair + campaign).even? ? %w[A B] : %w[B A]).map { |arm| [campaign, pair, procs, arm] }
      end
    end
  end
  raise 'Unexpected invocation order/count' unless batches.map { |b| b.values_at('campaign', 'pair', 'procs', 'arm') } == expected
  batches.each do |batch|
    expected_sha = phase == 'old-v2' && batch['arm'] == 'B' ? v2_sha : old_sha
    raise 'Wrong invocation binary' unless batch['hash'] == expected_sha
    raise 'Missing PASS or wrong ordered cells' unless batch['pass'] && batch['rows'].map { |r| r['benchmark'] } == NAMES
  end
  warn "Validated #{phase}: #{batches.length} invocations, #{batches.sum { |b| b['rows'].length }} records, fixed N=#{N}"
  batches
end

def metric_summary(a, b)
  deltas = a.zip(b).map { |x, y| y - x }
  [median(a), median(b), median(deltas), deltas.min, deltas.max,
   deltas.count { |v| v > 0 }, deltas.count(0), deltas.count { |v| v < 0 },
   rank_p(a, b), JSON.generate(deltas)]
end

raise 'usage: analyze.rb OLD_OLD OLD_V2 OLD_SHA256 V2_SHA256' unless ARGV.length == 4
old_old, old_v2, old_sha, v2_sha = ARGV
raise 'Invalid expected hashes' unless [old_sha, v2_sha].all? { |sha| sha.match?(/\A[0-9a-f]{64}\z/) }
groups = {}
[[old_old, 'old-old'], [old_v2, 'old-v2']].each do |path, phase|
  parse(path, phase, old_sha, v2_sha).each do |batch|
    batch['rows'].each do |row|
      key = [phase, batch['campaign'], batch['procs'], row['benchmark']]
      group = (groups[key] ||= { 'A' => {}, 'B' => {} })
      raise 'Duplicate pair' if group[batch['arm']].key?(batch['pair'])
      group[batch['arm']][batch['pair']] = row
    end
  end
end
raise 'Incomplete comparison cells' unless groups.length == 36
csv = CSV.new($stdout)
head = %w[phase campaign procs benchmark n]
%w[bytes allocs].each do |metric|
  head.concat(%w[A_median B_median paired_delta_median paired_delta_min paired_delta_max positive_pairs zero_pairs negative_pairs nominal_rank_p paired_deltas].map { |s| "#{metric}_#{s}" })
end
%w[bytes allocs].each { |m| %w[A B].each { |arm| %w[per_op remainder].each { |suffix| head << "#{arm}_#{m}_#{suffix}_by_pair" } } }
csv << head
groups.sort.each do |key, group|
  raise 'Incomplete pairs' unless group.values.all? { |arm| arm.keys.sort == (1..7).to_a }
  a, b = %w[A B].map { |arm| (1..7).map { |pair| group[arm].fetch(pair) } }
  row = key + [N]
  %w[bytes allocs].each { |metric| row.concat(metric_summary(a.map { |r| r["total_#{metric}"] }, b.map { |r| r["total_#{metric}"] })) }
  %w[bytes allocs].each { |metric| [a, b].each { |arm| %w[per_op remainder].each { |suffix| row << JSON.generate(arm.map { |r| r["#{metric}_#{suffix}"] }) } } }
  csv << row
end
warn 'All 36 comparison cells retained; every paired raw-total delta and quotient/remainder is reported.'
warn 'Descriptive diagnostic only: no equivalence assertion, noise subtraction, V2 rescoring, promotion or external leadership claim.'
