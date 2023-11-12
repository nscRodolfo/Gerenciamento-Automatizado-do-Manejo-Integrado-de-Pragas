package my.aplication.manejointeligentedepragas.model;

public class InimigoNaturalModel {

    private int Cod_Inimigo_Natural;
    private String Nome;
    private String Familia;
    private String Ordem;
    private String NomeCientifico;
    private String Descricao;

    public InimigoNaturalModel(int cod_Inimigo_Natural, String nome, String familia, String ordem, String nomeCientifico, String descricao) {
        this.Cod_Inimigo_Natural = cod_Inimigo_Natural;
        this.Nome = nome;
        this.Familia = familia;
        this.Ordem = ordem;
        this.NomeCientifico = nomeCientifico;
        this.Descricao = descricao;
    }

    public int getCod_Inimigo_Natural() {
        return Cod_Inimigo_Natural;
    }

    public void setCod_Inimigo_Natural(int cod_Inimigo_Natural) {
        this.Cod_Inimigo_Natural = cod_Inimigo_Natural;
    }

    public String getNome() {
        return Nome;
    }

    public void setNome(String nome) {
        this.Nome = nome;
    }

    public String getFamilia() {
        return Familia;
    }

    public void setFamilia(String familia) {
        this.Familia = familia;
    }

    public String getOrdem() {
        return Ordem;
    }

    public void setOrdem(String ordem) {
        this.Ordem = ordem;
    }

    public String getNomeCientifico() {
        return NomeCientifico;
    }

    public void setNomeCientifico(String nomeCientifico) {
        this.NomeCientifico = nomeCientifico;
    }

    public String getDescricao() {
        return Descricao;
    }

    public void setDescricao(String descricao) {
        this.Descricao = descricao;
    }
}
